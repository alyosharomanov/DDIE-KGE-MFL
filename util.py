# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import torch

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ColorHandler(logging.StreamHandler):
    # https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    GRAY8 = "38;5;8"
    GRAY7 = "38;5;7"
    ORANGE = "33"
    RED = "31"
    WHITE = "0"

    level_color_map = {
        logging.DEBUG: GRAY8,
        logging.INFO: GRAY7,
        logging.WARNING: ORANGE,
        logging.ERROR: RED,
    }

    csi = f"{chr(27)}["  # control sequence introducer

    def emit(self, record):
        try:
            color = self.level_color_map.get(record.levelno, self.WHITE)
            msg = f"{self.csi}{color}m{self.format(record)}{self.csi}m"
            stream = self.stream
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)
