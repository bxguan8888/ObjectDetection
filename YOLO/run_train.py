import mxnet as mx 
from Symbol.symbol import get_resnet_model
import numpy as np
from data_ulti import get_iterator
from tools.logging_metric import LogMetricsCallback, LossMetric
import time
import argparse

import logging
import sys
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # example cmd using cpu:
    # python run_train.py --cpu 1 --train_data_path DATA_rec/drive_small.rec --val_data_path DATA_rec/drive_small.rec --checkpoint_prefix models/drive_small_detect --checkpoint_interval 1 --batch_size 1 --epoch 5 --lambda_noobj 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', help='train_data_path', default="DATA_rec/drive_full.rec")
    parser.add_argument('--val_data_path', help='val_data_path', default="DATA_rec/drive_full.rec")
    parser.add_argument('--cpu', help='use cpu if set to 1', type=int, default=0)
    parser.add_argument('--checkpoint_prefix', help='checkpoint_prefix', default="models/drive_full_detect")
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='checkpoint_interval')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning_rate')
    parser.add_argument('--lambda_noobj', type=float, default=0.2, help='lambda_noobj')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')


    args = parser.parse_args()
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    checkpoint_prefix = args.checkpoint_prefix
    checkpoint_interval = args.checkpoint_interval
    batch_size = args.batch_size
    lambda_noobj = args.lambda_noobj
    learning_rate = args.learning_rate
    epoch = args.epoch
    cpu = args.cpu
    if cpu == 1:
        context = mx.cpu(0)
    else:
        context = mx.gpu(0)

    # get sym
    # Try different network 18, 50, 101 to find the best one
    sym = get_resnet_model('pretrained_models/resnet-34', 0, lambda_noobj)
    _, args_params, aux_params = mx.model.load_checkpoint('pretrained_models/resnet-34', 0)

    # get some input
    # change it to the data rec you create, and modify the batch_size
    train_data = get_iterator(path=train_data_path, data_shape=(3, 224, 224), label_width=7*7*9, batch_size=batch_size, shuffle=True)
    val_data = get_iterator(path=val_data_path, data_shape=(3, 224, 224), label_width=7*7*9, batch_size=batch_size)

    # allocate gpu/cpu mem to the sym
    mod = mx.mod.Module(symbol=sym, context=context)

    # setup metric
    # metric = mx.metric.create(loss_metric, allow_extra_outputs=True)
    tme = time.time()
    logtrain = LogMetricsCallback('logs/train_'+str(tme))

    # setup monitor for debugging
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = None #mx.mon.Monitor(10, norm_stat, pattern=".*backward*.")

    # save model
    checkpoint = mx.callback.do_checkpoint(checkpoint_prefix, checkpoint_interval)

    # Train
    # Try different hyperparamters to get the model converged, (batch_size,
    # optimization method, training epoch, learning rate/scheduler)
    mod.fit(train_data=train_data,
            eval_data=val_data,
            num_epoch=epoch,
            monitor=mon,
            eval_metric=LossMetric(0.5),
            optimizer='rmsprop',
            optimizer_params={'learning_rate':learning_rate, 'lr_scheduler': mx.lr_scheduler.FactorScheduler(300000, 0.1, 0.001)},
            initializer=mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'),
            arg_params=args_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=[mx.callback.Speedometer(batch_size=32, frequent=10, auto_reset=False), logtrain],
            epoch_end_callback=checkpoint
             )
