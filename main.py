import os 
import torch 
import time 
import argparse
from torch.cuda.amp import autocast, GradScaler
from datasets.nnunet_dl import get_dataset, get_tree_seg_dataloader
from utils.log import get_logger, AverageMeter
from utils.train_utils import dice_coefficient, load_checkponit, dummy_context, set_seed
from predict_utils import predict
from utils.gpu import set_gpu
from utils.parse import parse_yaml, format_config,write_json
from evaluation import eval

def run_iteration(args, net, optimizer, scaler, gen_data, iters, phase, label_type, loss1, loss2, loss3, device, warm_up, eps):
    loss_ind = AverageMeter()
    loss1_ind = AverageMeter()
    loss2_ind = AverageMeter()
    loss3_ind = AverageMeter()

    for batch in range(iters):
        data_batch = next(gen_data)
        data = data_batch['data']
        seg = data_batch['target'] 

        data = data.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        with torch.no_grad():
            pos_u = (seg == 1).type_as(data)
            pos_l = (seg == 2).type_as(data)

            if label_type == 'part':
                target = pos_l
            elif label_type == 'comp':
                target = pos_l + pos_u
            else:
                raise NotImplementedError(label_type)
            
        with torch.no_grad() if phase == 'val' else dummy_context():
            with autocast() if args.fp16 else dummy_context():
                p_y_condi_x_logits, p_s_condi_yx_logits = net(data)
                f_x, e_x = torch.sigmoid(p_y_condi_x_logits), torch.sigmoid(p_s_condi_yx_logits)
                s_x = f_x * e_x
                with torch.no_grad():
                    # net.eval()
                    # warm up
                    if warm_up :
                        p_y_condi_sx = target
                        unlabeled = f_x 
                    else:
                        # P(y|s,x)
                        unlabeled = f_x * (1 - e_x)/(1 - f_x*e_x + eps)                        
                        p_y_condi_sx = target + (1-target) * unlabeled
               
                l1 = loss1(p_y_condi_x_logits, p_y_condi_sx)
                if 'head' in args.mlp:
                    l2 = loss2(p_s_condi_yx_logits, target, p_y_condi_sx)  if not warm_up else torch.zeros((1,)).cuda()
                else:
                    l2 = loss2(p_s_condi_yx_logits, target, p_y_condi_sx)  if not warm_up else loss1(p_s_condi_yx_logits, target)
                
                l3 = loss3(s_x, target) if loss3 is not None and not warm_up else torch.zeros((1,)).cuda()

                l = l1 + args.lambda_*l2

                if loss3 is not None:
                    l += args.mu_*l3

            batch_size = data.shape[0]

            if phase == 'train':
                optimizer.zero_grad()
                if args.fp16:
                    scaler.scale(l).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    l.backward()
                    optimizer.step()

        loss_ind.update(l.item(), batch_size)
        loss1_ind.update(l1.item(), batch_size)
        loss2_ind.update(l2.item(), batch_size)
        loss3_ind.update(l3.item(), batch_size)

    return loss_ind, loss1_ind, loss2_ind, loss3_ind

def main(config, args):
    batch_size = config['batch_size']
    patch_size = config['patch_size']
    oversample = config['oversample']
    iters_train = config['iters_train']
    iters_val = config['iters_val']
    num_processes = config['num_processes']
    init_num_channels = config['init_num_channels']
    do_ds = config['do_ds']
    num_epochs = config['num_epochs'] + args.warm_up
    label_type = config['label_type']

    input_channel = 1
    num_classes = 1
    set_seed(42)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True 
    device = torch.device('cuda')

    # network
    from nets.unet import nnUNet3D_branch,nnUNet3D_DualNet,nnUNet3D_MLPs

    if args.mlp == 'head':
        net = nnUNet3D_MLPs(input_channel, num_classes, init_num_channels, do_ds, dims=args.mlp_dim)
    elif args.mlp == 'branch':
        net = nnUNet3D_branch(input_channel, num_classes, init_num_channels, do_ds)
    elif args.mlp == 'net':
        net = nnUNet3D_DualNet(input_channel, num_classes, init_num_channels, do_ds)

    net = net.cuda()

    # dataset
    ds_train = get_dataset(config['train_file'], config['data_dir'])
    ds_val = get_dataset(config['val_file'], config['data_dir'])

    # dataloader
    gen_train, gen_val = get_tree_seg_dataloader(ds_train, ds_val, part_type=args.part_type, dist_type=None, 
                                                patch_size=patch_size, batch_size=batch_size, iters_train=iters_train,
                                                iters_val=iters_val, oversample=oversample, num_processes=num_processes, 
                                                mirror_axes=None, win_min=config['win_min'], win_max=config['win_max'])

    # loss 
    eps = 1e-5 if args.fp16 else 1e-12
    from loss.pu_loss import LPE_theta1_loss, LPE_theta2_loss
    from loss.loss_func import LIB_lossv2
    loss1 = LPE_theta1_loss()
    loss2 = LPE_theta2_loss()
    loss3 = None

    if args.loss == 'gul':
        from loss.loss_func import LIB_lossv2
        loss3 = LIB_lossv2(alpha=args.alpha, final_nonlin=lambda x:x)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    def lr_lambda(epoch): return (1 - float(epoch)/(num_epochs - args.warm_up))**0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda
    )

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S_", time.localtime())
    timetag = timestamp+config['tag']
    test_dir = os.path.join("results", timetag, 'prediction')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    ckpt_path = os.path.join("results", timetag, "ckpt")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    write_json(config, os.path.join('results', timetag, "config.json"), verbose=True)
    log_path = os.path.join("results", timetag, "log.txt")
    logger = get_logger(log_path)

    logger.info(">>>The config is: \n")
    logger.info(format_config(config))

    if args.fp16:
        scaler = GradScaler()
    else:
        scaler = None 
    
    best_result = {
        'loss': 999,
        'epoch': 0,
    }
    
    for epoch in range(num_epochs):
        # train
        net.train()
        s_time = time.time()
        loss_ind, loss1_ind, loss2_ind, loss3_ind = run_iteration(args, net, optimizer, scaler, gen_train, iters_train, 'train', label_type, loss1, loss2, loss3, device, epoch < args.warm_up, eps)
            
        if epoch >= args.warm_up :
            lr_scheduler.step()

        logger.info('train epoch:{}, loss:{:.4f}, loss1:{:.4f}, loss2:{:.4f}, loss3:{:.4f}, {:.1f} s'.format(epoch, loss_ind.avg, loss1_ind.avg, loss2_ind.avg, loss3_ind.avg, time.time()-s_time)) 

        # validation
        net.eval()
        s_time = time.time()
        loss_ind, loss1_ind, loss2_ind, loss3_ind = run_iteration(args, net, optimizer, scaler, gen_val, iters_val, 'val', label_type, loss1, loss2, loss3, device, epoch < args.warm_up, eps)

        logger.info('val epoch:{}, loss:{:.4f}, loss1:{:.4f}, loss2:{:.4f}, loss3:{:.4f}, {:.1f} s'.format(epoch, loss_ind.avg, loss1_ind.avg, loss2_ind.avg, loss3_ind.avg, time.time()-s_time)) 
    

        if epoch >= args.warm_up:
            if loss_ind.avg < best_result['loss']:
                best_result['loss'] = loss_ind.avg
                best_result['epoch'] = epoch

                states = {
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, os.path.join(ckpt_path, "model_best.pth"))

        if epoch == num_epochs -1 :
            states = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(states, os.path.join(ckpt_path, "model_last.pth"))

    logger.info('best loss:{:.4f}({})'.format(
            best_result['loss'], best_result['epoch']))
        
    gen_train._finish()
    gen_val._finish()
    
    ds_test = get_dataset(config['test_file'], config['data_dir'])

    net = load_checkponit(net, ckpt_path, model_name='model_best.pth')
    pre_dir = os.path.join(test_dir, 'best')
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    predict(net, ds_test, num_classes, patch_size, pre_dir, step_size=0.5, mirror_axes=None, win_min=config['win_min'], win_max=config['win_max'], th=0.5, over_write=True)
    eval(pre_dir, config['label_dir'], os.path.join('results', ckpt_path.split(os.sep)[-2]),
         num_classes=num_classes, metrics_str=config['metrics_str'], file_name='result_best', label_name=config['label_name'], th=0.5)
    
    net = load_checkponit(net, ckpt_path, model_name='model_last.pth')
    pre_dir = os.path.join(test_dir, 'model_last')
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    predict(net, ds_test, num_classes, patch_size, pre_dir, step_size=0.5, mirror_axes=None, win_min=config['win_min'], win_max=config['win_max'], th=0.5, over_write=True)
    eval(pre_dir, config['label_dir'], os.path.join('results', ckpt_path.split(os.sep)[-2]),
        num_classes=num_classes, metrics_str=config['metrics_str'], file_name='result_last', label_name=config['label_name'], th=0.5)

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

if __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='AirwaySeg')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--config', type=str, default='config/cfg.yaml')
    parser.add_argument('--ds', type=str, default='BAS')
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--label_type', type=str, default='comp')
    parser.add_argument('--part_type', type=str, default=None)
    parser.add_argument('--lambda_', type=float, default=1.)
    parser.add_argument('--mu_', type=float, default=0.)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--mlp', type=str, default='head')
    parser.add_argument('--mlp_dim', type=list_of_ints)
    parser.add_argument('--postprop', type=str, default='LCC')

    args = parser.parse_args()
    gpus = set_gpu(args.gpu)

    config = parse_yaml(args.config)

    dataset = args.ds 

    print(args.gpu, dataset)
    
    config['label_name'] ='airway_label.nii.gz'
    root_dir = config['npy_dir']
    config['data_dir'] = f'{root_dir}/data/{dataset}_prep'

    config['label_dir'] = f"{config['label_dir']}/{dataset}"
    config['train_file'] = f'training_files/TreeSeg_{dataset}/train_ids.pkl'
    config['val_file'] = f'training_files/TreeSeg_{dataset}/val_ids.pkl'
    config['test_file'] = f'training_files/TreeSeg_{dataset}/test_ids.pkl'

    config['tag'] = 'tree_seg_nnUNet3D_LPE-cl_{}_{}_lambda_-{}_mu_-{}_aug_amp-{}_warm-up-{}'.format(args.label_type, args.part_type, args.lambda_, args.mu_, dataset, args.fp16, args.warm_up)
    config['amp'] = args.fp16
    config['label_type'] = args.label_type
    config['part_type'] = args.part_type
    config['mlp'] = args.mlp
    config['loss'] = args.loss
    config['alpha'] = args.alpha
    config['mlp_dim'] = args.mlp_dim
    config['lambda_'] = args.lambda_
    config['mu_'] = args.mu_
    
    if args.resume is not None:
        config['resume'] = os.path.join("results", args.resume, "ckpt")

    if args.infer  is not None:
        config['ckpt_path'] = os.path.join("results", args.infer, "ckpt")
        config['test_dir'] = os.path.join("results", args.infer, "prediction")

    main(config, args)
