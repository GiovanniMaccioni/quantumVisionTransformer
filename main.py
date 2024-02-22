import argparse
import training_pipe

config ={}

if __name__=="__main__":

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description='QVT')
    parser.add_argument("--dir_path", type=str, default="./temp", help='')
    parser.add_argument("--model", type=str, default="qvt", help='')
    parser.add_argument("--hybrid", default=False, action='store_true', help='')
    parser.add_argument("--epochs", type=int, default=20, help='epochs')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
    parser.add_argument("--lr", type=float, default=0.003, help='')
    parser.add_argument("--classes", type=int, nargs='+', default=[0,1], help='Chosen classes')


    #Arguments cnn
    parser.add_argument("--embed_dim", type=int, default=4, help='embed_dim')
    parser.add_argument("--hidden_dim", type=int, default=32, help='')
    parser.add_argument("--num_heads", type=int, default=8, help='')
    parser.add_argument("--num_layers", type=int, default=1, help='')
    parser.add_argument("--patch_size", type=int, default=14, help='')
    parser.add_argument("--num_channels", type=int, default=1, help='')
    parser.add_argument("--num_patches", type=int, default=4, help='')
    #parser.add_argument("--num_classes", type=int, default=32, help='')
    parser.add_argument("--dropout", type=float, default=0.2, help='')

    
    parser.add_argument("--vec_loader", type=str, default='diagonal', help='')
    parser.add_argument("--ort_layer", type=str, default='pyramid', help='')

    args = parser.parse_args()

    config['dir_path']=args.dir_path
    config['model'] = args.model
    config['hybrid'] = args.hybrid
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['classes'] = args.classes

    config['embed_dim'] = args.embed_dim
    config['hidden_dim'] = args.hidden_dim
    config['num_channels'] = args.num_channels
    config['num_heads'] = args.num_heads
    config['num_layers'] = args.num_layers
    config['num_classes'] = len(args.classes)
    config['patch_size'] = args.patch_size
    config['num_patches'] = args.num_patches
    config['dropout'] = args.dropout

    config['vec_loader'] = args.vec_loader
    config['ort_layer'] = args.ort_layer

    training_pipe.run(config)