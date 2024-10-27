from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import functools
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import os
from copy import deepcopy
import time
#import common.metrics
#import models


def update_argparser(parser):
  models.update_argparser(parser)
  args, _ = parser.parse_known_args()
  parser.add_argument(
      '--num_blocks',
      help='Number of residual blocks in networks.',
      default=16,
      type=int)
  parser.add_argument(
      '--num_residual_units',
      help='Number of residual units in networks.',
      default=32,
      type=int)
  parser.add_argument(
      '--width_multiplier',
      help='Width multiplier inside residual blocks.',
      default=4,
      type=float)
  parser.add_argument(
      '--temporal_size',
      help='Number of frames for burst input.',
      default=None,
      type=int)
  if args.dataset.startswith('vsd4k'):
    parser.set_defaults(
        train_epochs=50,
        learning_rate_milestones=(20,25),
        learning_rate_decay=0.1,
        save_checkpoints_epochs=1,
        train_temporal_size=1,
        eval_temporal_size=1,
    )
  else:
    raise NotImplementedError('Needs to tune hyper parameters for new dataset.')


# def get_model_spec(params):
#   model = MODEL(params)
#   print('# of parameters: ', sum([p.numel() for p in model.parameters()]))
#   optimizer = optim.Adam(model.parameters(), params.learning_rate)
#   lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=params.train_epochs,eta_min=4e-08)
#   loss_fn = torch.nn.L1Loss()
#   metrics = {
#       'loss':
#           loss_fn,
#       'PSNR':
#           functools.partial(
#               common.metrics.psnr,
#               shave=0 if params.scale == 1 else params.scale +4),
#       # ,
#       'PSNR_MSE':
#           functools.partial(
#               common.metrics.psnr_mse,
#               shave=0 if params.scale == 1 else params.scale+4),
#   }
#   return model, loss_fn, optimizer, lr_scheduler, metrics

class Block(nn.Module):

  def __init__(self,
               num_residual_units,
               kernel_size,
               width_multiplier=1,
               weight_norm=torch.nn.utils.weight_norm,
               res_scale=1):
    super(Block, self).__init__()
    body = []
    conv = weight_norm(
        nn.Conv2d(
            num_residual_units,
            int(num_residual_units * width_multiplier),
            kernel_size,
            padding=kernel_size // 2))
    init.constant_(conv.weight_g, 2.0)
    init.zeros_(conv.bias)
    body.append(conv)
    body.append(nn.ReLU(True))
    conv = weight_norm(
        nn.Conv2d(
            int(num_residual_units * width_multiplier),
            num_residual_units,
            kernel_size,
            padding=kernel_size // 2))
    init.constant_(conv.weight_g, res_scale)
    init.zeros_(conv.bias)
    body.append(conv)

    self.body = nn.Sequential(*body)

  def forward(self, x):
    x = self.body(x) + x
    return x

class MODEL(nn.Module):

  def __init__(self, opt):
    super(MODEL, self).__init__()
    self.temporal_size = opt.temporal_size # None
    self.image_mean = opt.image_mean # 0.5
    kernel_size = 3
    skip_kernel_size = 5
    weight_norm = torch.nn.utils.weight_norm
    num_inputs = opt.num_channels #3
    if self.temporal_size:
      num_inputs *= self.temporal_size
    self.scale = opt.sr_ratio
    num_outputs = opt.sr_ratio * opt.sr_ratio * opt.num_channels

    body = []
    conv = weight_norm(
        nn.Conv2d(
            num_inputs,
            opt.num_residual_units,
            kernel_size,
            padding=kernel_size // 2))
    init.ones_(conv.weight_g)
    init.zeros_(conv.bias)
    body.append(conv)
    for _ in range(opt.num_blocks):
      body.append(
          Block(
              opt.num_residual_units,
              kernel_size,
              opt.width_multiplier,
              weight_norm=weight_norm,
              res_scale=1 / math.sqrt(opt.num_blocks),
          ))
    conv = weight_norm(
        nn.Conv2d(
            opt.num_residual_units,
            num_outputs,
            kernel_size,
            padding=kernel_size // 2))
    init.ones_(conv.weight_g)
    init.zeros_(conv.bias)
    body.append(conv)
    self.body = nn.Sequential(*body)

    skip = []
    if num_inputs != num_outputs:
      conv = weight_norm(
          nn.Conv2d(
              num_inputs,
              num_outputs,
              skip_kernel_size,
              padding=skip_kernel_size // 2))
      init.ones_(conv.weight_g)
      init.zeros_(conv.bias)
      skip.append(conv)
    self.skip = nn.Sequential(*skip)

    shuf = []
    if opt.sr_ratio > 1:
      shuf.append(nn.PixelShuffle(opt.sr_ratio))
    self.shuf = nn.Sequential(*shuf)

  def forward(self, x, cond = None):
    #print('temporal_size and image_mean', self.temporal_size,self.image_mean) # None 0.5
    if self.temporal_size:
      x = x.view([x.shape[0], -1, x.shape[3], x.shape[4]])
    x -= self.image_mean
    x = self.body(x) + self.skip(x)
    x = self.shuf(x)
    x += self.image_mean
    if self.temporal_size:
      x = x.view([x.shape[0], -1, 1, x.shape[2], x.shape[3]])
    return x
  

  def load_network(self, load_path, device, strict=True, param_key='params_ema'):
      """Load network.

      Args:
          load_path (str): The path of networks to be loaded.
          net (nn.Module): Network.
          strict (bool): Whether strictly loaded.
          param_key (str): The parameter key of loaded network. If set to
              None, use the root 'path'.
              Default: 'params'.
      """
      # net = self.get_bare_model(net)
      load_net = torch.load(load_path, map_location=device)
      if param_key is not None:
          if param_key not in load_net and 'params' in load_net:
              param_key = 'params'
              print('Loading: params_ema does not exist, use params.')
          load_net = load_net[param_key]
      print(f'Loading {self.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
      # remove unnecessary 'module.'
      for k, v in deepcopy(load_net).items():
          if k.startswith('module.'):
              load_net[k[7:]] = v
              load_net.pop(k)
      self._print_different_keys_loading(load_net, strict)
      self.load_state_dict(load_net, strict=strict)

  def _print_different_keys_loading(self, load_net, strict=True):
      """Print keys with differnet name or different size when loading models.

      1. Print keys with differnet names.
      2. If strict=False, print the same key but with different tensor size.
          It also ignore these keys with different sizes (not load).

      Args:
          crt_net (torch model): Current network.
          load_net (dict): Loaded network.
          strict (bool): Whether strictly loaded. Default: True.
      """
      crt_net = self.state_dict()
      crt_net_keys = set(crt_net.keys())
      load_net_keys = set(load_net.keys())

      if crt_net_keys != load_net_keys:
          print('Current net - loaded net:')
          for v in sorted(list(crt_net_keys - load_net_keys)):
              print(f'  {v}')
          print('Loaded net - current net:')
          for v in sorted(list(load_net_keys - crt_net_keys)):
              print(f'  {v}')

      # check the size for the same keys
      if not strict:
          common_keys = crt_net_keys & load_net_keys
          for k in common_keys:
              if crt_net[k].size() != load_net[k].size():
                  print(f'Size different, ignore [{k}]: crt_net: '
                                  f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                  load_net[k + '.ignore'] = load_net.pop(k)

  def save_network(self, save_root, net_label, current_iter, param_key='params'):

      if current_iter == -1:
          current_iter = 'latest'
      save_filename = f'{net_label}_{current_iter}.pth'
      save_path = os.path.join(save_root, save_filename)

      net = self if isinstance(self, list) else [self]
      param_key = param_key if isinstance(param_key, list) else [param_key]
      assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

      save_dict = {}
      for net_, param_key_ in zip(net, param_key):
          state_dict = net_.state_dict()
          for key, param in state_dict.items():
              if key.startswith('module.'):  # remove unnecessary 'module.'
                  key = key[7:]
              state_dict[key] = param.cpu()
          save_dict[param_key_] = state_dict

      # avoid occasional writing errors
      retry = 3
      while retry > 0:
          try:
              torch.save(save_dict, save_path)
          except Exception as e:
              print(f'Save model error: {e}, remaining retry times: {retry - 1}')
              time.sleep(1)
          else:
              break
          finally:
              retry -= 1
      if retry == 0:
          print(f'Still cannot save {save_path}. Just ignore it.')

  def tile_process(self, img, cond, tile_size, tile_pad=10):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)
        cond = cond.unsqueeze(0)

        # start with black image
        output = img.new_zeros(output_shape).to('cpu')
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        
        input_show = img[:, :, 0:138, 118:266]
        cond_show = cond[:, :, 0:138, 118:266]        
        with torch.no_grad():
            out_show = self(input_show, cond_show)
            pred_show = out_show.squeeze().movedim(0, -1).detach().clamp(0, 1).cpu().numpy()
            cv2.imwrite(f'trial/ObamaSR_tile32/tile/show.png', cv2.cvtColor((pred_show * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                cond_tile = cond[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                # upscale tile
                # try:
                with torch.no_grad():
                    output_tile = self(input_tile,cond_tile)#, cond_tile
                    #print(f'input_tile:{input_tile.shape}, output_tile:{output_tile.shape}')
                # except Exception as error:
                #     print('Error', error)
                #print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')
                    print(f'input--> input_start_y_pad{input_start_y_pad}:input_end_y_pad{input_end_y_pad}, input_start_x_pad{input_start_x_pad}:input_end_x_pad{input_end_x_pad}')
                    pred_sr = output_tile.squeeze().movedim(0, -1).detach().clamp(0, 1).cpu().numpy()
                    cv2.imwrite(f'trial/ObamaSR_tile32/tile/{x}_{y}.png', cv2.cvtColor((pred_sr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale
                print(f'output_tile shape:{output_tile.shape},output_start_x_tile:{output_start_x_tile}, output_end_x_tile:{output_end_x_tile}')
                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile].detach().to('cpu')
        return output