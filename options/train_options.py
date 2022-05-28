from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training parameters
        parser.add_argument('--no_amp', type=bool, default=False, help="Do not use mixed precision training.")
        parser.add_argument('--sync_bn', type=bool, default=False, help="Synchronize BatchNorm across all devices. Use when batchsize is small.")
        parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=200, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay for AdamW.')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        
        parser.add_argument('--use_wandb', type=bool, default=True, help="Use wandb logger.")
        parser.add_argument('--project', type=str, default='nerf_refine', help="Name of wandb project.")
        parser.add_argument('--resume', type=str, default=None, help="ID of wandb run to resume.")
        parser.add_argument('--init_epoch', type=int, default=1, metavar="N", help="Initial epoch.")
        parser.add_argument('--log_interval', type=int, default=20, metavar="N", help="Log per N steps.")
        parser.add_argument('--img_log_interval', type=int, default=300, metavar="N", help="Log images per N steps.")

        self.isTrain = True
        return parser
