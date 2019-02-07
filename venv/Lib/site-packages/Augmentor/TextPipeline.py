# TextPipeline.py
# Author: Marcus D. Bloice <https://github.com/mdbloice> and contributors
# Licensed under the terms of the MIT Licence.
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


class TextPipeline(object):

    def __init__(self, textfile, labels):

        line_count = 0

        with open(textfile, 'r') as file:
            all_lines = file.readlines()
            line_count += 1

        if line_count == len(labels):
            return "Init complete."
        else:
            return 0
