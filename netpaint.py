#! /usr/bin/env python3

# This file is a part of NetPaint (http://github.com/SyntheticDreams/NetPaint)
#
#    Copyright (C) 2019 Synthetic Dreams (Anthony Westbrook)
#
#    This program is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

import argparse
import math
import os
import struct
import sys
import time
import urwid
from functools import partial
from PIL import Image 

class WidgetManager():
    """ Manage all widgets and associated functionality """
    def __init__(self, palette, encoding):
        self.palette = palette
        self.encoding = encoding
        self.widgets = dict()
        self.groups = dict()
        self.base = None

        self.update_palette()

    def register(self, name, widget, attr=None):
        """ Register new widget """
        self.widgets[name] = urwid.AttrMap(widget, attr)

    def get(self, name, orig=False):
        """ Get a widget attr or base object """
        if orig:
            return self.widgets[name]._original_widget
        else:
            return self.widgets[name]

    def __setitem__(self, key, value):
        """ Overload for register shortcut """
        self.register(key, value)

    def __getitem__(self, key):
        """ Overload for get shortcut """
        return self.get(key)

    def update_attr(self, name, attr):
        """ Update attribute associated with widget """
        self.widgets[name].set_attr_map({None: attr})

    def group_add(self, group, objects, multi=False):
        """ Add object (widget/tuples/other) to a group """
        self.groups.setdefault(group, [])

        if multi:
            self.groups[group].extend(objects)
        else:
            self.groups[group].append(objects)

    def group_get(self, group):
        """ Get object list for a group """
        self.groups.setdefault(group, [])

        return self.groups[group]

    def register_base(self, base, default):
        """ Register overlay base and default child """
        self.base = (base, default)

    def activate_overlay(self, name):
        """ Activate overlay over canvas portion of body """
        if isinstance(self.get(self.base[0], True).contents[0][0]._original_widget, urwid.Overlay):
            return

        # Reset overlay if applicable
        if getattr(self.get(name, True), "reset", False):
            self.get(name, True).reset()

        self.get(self.base[0], True).contents[0] = (self.widgets[name], self.get(self.base[0], True).contents[0][1])

    def deactivate_overlay(self):
        """ Deactivate overlay covering canvas """
        if isinstance(self.get(self.base[0], True).contents[0][0]._original_widget, urwid.Overlay):
            self.get(self.base[0], True).contents[0] = (self.widgets[self.base[1]], self.get(self.base[0], True).contents[0][1])

    def get_overlay(self):
        """ Get current overlay """
        if not isinstance(self.get(self.base[0], True).contents[0][0]._original_widget, urwid.Overlay):
            return None

        return self.get(self.base[0], True).contents[0][0]._original_widget

    def get_color_name(self, color, clean=True):
        """ Generate the text color name for the index """
        name = urwid.display_common._BASIC_COLORS[color]
        if clean:
            name = name.replace(" ", "_")

        return name

    def update_palette(self):
        """ Add color name combinations to palette """
        for fore_idx in range(16):
            for back_idx in range(16):
                colors_id = "{}-{}".format(self.get_color_name(fore_idx), self.get_color_name(back_idx))
                colors = (colors_id, self.get_color_name(fore_idx, False), self.get_color_name(back_idx, False))
                self.palette.append(colors)


class TextGrid(urwid.Widget):
    """ Provide coordinate addressable text grid """
    CONFIG_DEFAULTS = {"back_char": b" ", "back_attrs": (0, 0, 0), "ext_char": bytes("\u2591", "utf8"), "ext_attrs": (8, 0, 0),
                       "scroll_attrs": (0, 7, 0), "dims": None, "scroll_h": True, "scroll_v": True}
    _sizing = frozenset([urwid.BOX])

    def __init__(self, wm, config={}):
        self.wm = wm
        self.config = dict(self.CONFIG_DEFAULTS)
        self.config.update(config)
        self.scroll_x = 0
        self.scroll_y = 0
        self.layer = 0

        # Initialize canvas
        self.reset()

    def convert_attrs(self, attrs):
        """ Convert TextGrid attributes to Urwid palette attribute """
        return "{}-{}".format(self.wm.get_color_name(attrs[0]), self.wm.get_color_name(attrs[1]))

    def plot(self, col, row, char, attrs, layer=None):
        """ Assign a character to a location on the grid """
        # Select desired layer
        if layer is None:
            layer = self.layer

        # Adjust dimensions for dynamic sized grids if necessary
        while row >= len(self.content[layer]):
            if self.config["dims"]:
                return

            self.content[layer].append([None] * len(self.content[layer][0]))
            self.attrs[layer].append([None] * len(self.content[layer][0]))

        while col >= len(self.content[layer][0]):
            if self.config["dims"]:
                return

            for cur_row in range(len(self.content[layer])):
                self.content[layer][cur_row].append(None)
                self.attrs[layer][cur_row].append(None)

        # Assign character
        self.content[layer][row][col] = char
        self.attrs[layer][row][col] = attrs
        self._invalidate()

    def reset(self):
        """ Reset grid back to default """
        cols, rows = self.config["dims"] if self.config["dims"] else (1, 1)

        # Remove all layers and create bottom layer
        self.content = [[[None for col_idx in range(cols)] for row_idx in range(rows)]]
        self.attrs = [[[None for col_idx in range(cols)] for row_idx in range(rows)]]

        # Create cursor layer
        self.content.append([[None for col_idx in range(cols)] for row_idx in range(rows)])
        self.attrs.append([[None for col_idx in range(cols)] for row_idx in range(rows)])

        # Set bottom and cursor layer to visible
        self.visibility = [True, True]

        self._invalidate()

    def clear(self, layer=None):
        """ Clear the specified layer """
        # Select desired layer
        if layer is None:
            layer = self.layer

        for row_idx in range(len(self.content[layer])):
            for col_idx in range(len(self.content[layer][row_idx])):
                self.content[layer][row_idx][col_idx] = None
                self.attrs[layer][row_idx][col_idx] = None

        self._invalidate()

    def render(self, size, focus=False):
        """ Widget render implementation """
        encoding = self.wm.encoding

        # Calculate render dimensions (exclusive range)
        last_row = size[1] if len(size) == 2 else len(self.content[0])
        last_col = size[0] if len(size) > 0 else len(self.content[0][0])

        # Flatten layers into buffer
        flat_text, flat_attrs = self.get_flattened()

        # Copy rendered portion of flattened buffer
        text = [b"".join(row[self.scroll_x:last_col + self.scroll_x]) for row in flat_text[self.scroll_y:last_row + self.scroll_y]]

        attrs = []
        for row_idx in range(self.scroll_y, min(len(flat_attrs), last_row + self.scroll_y)):
            attrs.append([])
            for col_idx in range(self.scroll_x, min(len(flat_attrs[row_idx]), last_col + self.scroll_x)):
                cur_attr = (self.convert_attrs(flat_attrs[row_idx][col_idx]), len(flat_text[row_idx][col_idx]))
                attrs[row_idx - self.scroll_y].append(cur_attr)

        # Fill in remaining with background (dynamic) or invalid (fixed) characters
        fill_char = self.config["ext_char"] if self.config["dims"] else self.config["back_char"]
        fill_attrs = self.config["ext_attrs"] if self.config["dims"] else self.config["back_attrs"]

        while last_row > len(text):
            text.append(fill_char * len(text[0].decode(self.wm.encoding)))
            attrs.append([(self.convert_attrs(fill_attrs), len(fill_char))] * len(text[0].decode(encoding)))

        while last_col > len(text[0].decode(encoding)):
            for cur_row in range(last_row):
                if last_col > len(text[cur_row].decode(encoding)):
                    add_len = last_col - len(text[cur_row].decode(encoding))
                    text[cur_row] += (fill_char * add_len)
                    attrs[cur_row].extend([(self.convert_attrs(fill_attrs), len(fill_char))] * add_len)

        # If scrollbars are active, overwrite right/bottom edge
        if self.config["scroll_h"]:
            last_h = last_col - 3 if self.config["scroll_v"] else last_col - 2
            text[last_row - 1] = bytes("\u2190", "utf8") + bytes("\u2591", "utf8") * last_h + bytes("\u2192", "utf8")
            attrs[last_row - 1] = [(self.convert_attrs(self.config["scroll_attrs"]), len(char.encode(encoding))) for char in text[last_row - 1].decode(encoding)]

        if self.config["scroll_v"]:
            last_v = -2 if self.config["scroll_h"] else -1
            text[0] = text[0].decode(encoding)[:-1].encode(encoding) + bytes("\u2191", "utf8")
            attrs[0][-1] = (self.convert_attrs(self.config["scroll_attrs"]), 3)
            text[last_v] = text[last_v].decode(encoding)[:-1].encode(encoding) + bytes("\u2193", "utf8")
            attrs[last_v][-1] = (self.convert_attrs(self.config["scroll_attrs"]), 3)

            for (idx, row) in enumerate(text[1:last_v]):
                text[idx + 1] = row.decode(encoding)[:-1].encode(encoding) + bytes("\u2591", "utf8")
                attrs[idx + 1][-1] = (self.convert_attrs(self.config["scroll_attrs"]), 3)

        canvas = urwid.TextCanvas(text, attr=attrs, maxcol=last_col)

        return canvas

    def mouse_event(self, size, event, button, col, row, focus):
        """ Handle mouse actions (scrolling) """
        x = 0
        y = 0
        v_pos = size[1] - 2 if self.config["scroll_h"] else size[1] - 1
        h_pos = size[0] - 2 if self.config["scroll_v"] else size[0] - 1

        # Calculate scroll distance
        distance = 0

        if int(button) == 1 or int(button) == 4 or int(button) == 5:
            distance = 1
        if int(button) == 3:
            distance = 10

        # Detect if scroll buttons/bar clicked (bar currently does nothing)
        scroll = False

        if self.config["scroll_v"]:
            if col == size[0] - 1:
                scroll = True
                if row == 0:
                    # Scroll up
                    y = -distance
                if row == v_pos:
                    # Scroll down
                    y = distance

            # Scroll wheel
            if int(button) == 4:
                # Scroll up
                scroll = True
                y = -distance
            if int(button) == 5:
                # Scroll down
                scroll = True
                y = distance

        if self.config["scroll_h"]:
            if row == size[1] - 1:
                scroll = True
                if col == 0:
                    # Scroll left
                    x = -distance
                if col == h_pos:
                    # Scroll right
                    x = distance

        if scroll:
            self.scroll(x, y)
        else:
            # If unhandled, pass to grid event
            grid_col = col + self.scroll_x
            grid_row = row + self.scroll_y
            self.grid_event(size, event, button, grid_col, grid_row, focus)

    def grid_event(self, size, event, button, col, row, focus):
        """ Handle grid mouse action """
        pass

    def select_layer(self, idx):
        """ Change active layer """
        if (idx < 0) or (idx >= len(self.content) - 1):
            return False

        self.layer = idx

        return True

    def show_layer(self, idx, visible):
        """ Set layer visibility """
        if (idx < 1) or (idx >= len(self.content) - 1):
            return False

        self.visibility[idx] = visible

        return True

    def add_layer(self, idx):
        """ Add new empty layer """
        if (idx < 1) or (idx > len(self.content) - 1):
            return False

        rows = len(self.content[0])
        cols = len(self.content[0][0])

        self.content.insert(idx, [[None for col_idx in range(cols)] for row_idx in range(rows)])
        self.attrs.insert(idx, [[None for col_idx in range(cols)] for row_idx in range(rows)])
        self.visibility.insert(idx, True)

        return True

    def del_layer(self, idx):
        """ Delete an existing layer """
        if (idx < 1) or (idx >= len(self.content) - 1):
            return False

        del self.content[idx]
        del self.attrs[idx]
        del self.visibility[idx]
        self._invalidate()

        return True

    def scroll(self, x=0, y=0, rel=True):
        """ Scroll canvas in requested direction """
        if rel:
            self.scroll_x += x
            self.scroll_y += y
        else:
            self.scroll_x = 0
            self.scroll_y = 0

        # Check for boundaries
        if self.scroll_x < 0:
            self.scroll_x = 0
        if self.scroll_x >= len(self.content[0][0]):
            self.scroll_x = len(self.content[0][0]) - 1

        if self.scroll_y < 0:
            self.scroll_y = 0
        if self.scroll_y >= len(self.content[0]):
            self.scroll_y = len(self.content[0]) - 1

        # Update canvas
        self._invalidate()

    def get_flattened(self):
        """ Get flattened buffer """
        # Start with base of background character and color
        cols = len(self.content[0][0])
        rows = len(self.content[0])
        flat_text = [[self.config["back_char"] for col_idx in range(cols)] for row_idx in range(rows)]
        flat_attrs = [[self.config["back_attrs"] for col_idx in range(cols)] for row_idx in range(rows)]

        for layer_idx in range(len(self.content)):
            # Skip hidden layers
            if not self.visibility[layer_idx]:
                continue

            for row_idx in range(len(self.content[layer_idx])):
                for col_idx in range(len(self.content[layer_idx][row_idx])):
                    char = self.content[layer_idx][row_idx][col_idx]
                    attr = self.attrs[layer_idx][row_idx][col_idx]

                    if char is not None:
                        flat_text[row_idx][col_idx] = char

                    if attr is not None:
                        flat_attrs[row_idx][col_idx] = attr

        return flat_text, flat_attrs

    def get_dims(self):
        """ Get (layers, cols, rows) dimensions of image """
        layers = len(self.content)
        rows = len(self.content[0])
        cols = len(self.content[0][0])

        return layers, cols, rows

    def set_dims(self, cols, rows):
        """ Set dimensions of all layers """
        if cols < 1 or rows < 1:
            return False

        # Reset scroll to prevent offscreen draw
        self.scroll(0, 0, rel=False)
        cur_layers, cur_cols, cur_rows = self.get_dims()

        for layer_idx in range(cur_layers):
            if rows < cur_rows:
                # Truncate rows
                self.content[layer_idx] = self.content[layer_idx][:rows]
                self.attrs[layer_idx] = self.attrs[layer_idx][:rows]
            else:
                # Add rows
                self.content[layer_idx].extend([[None for ins_idx2 in range(cur_cols)] for ins_idx1 in range(rows - cur_rows)])
                self.attrs[layer_idx].extend([[None for ins_idx2 in range(cur_cols)] for ins_idx1 in range(rows - cur_rows)])

            for row_idx in range(rows):
                if cols < cur_cols:
                    # Truncate columns
                    self.content[layer_idx][row_idx] = self.content[layer_idx][row_idx][:cols]
                    self.attrs[layer_idx][row_idx] = self.attrs[layer_idx][row_idx][:cols]
                else:
                    # Add columns
                    self.content[layer_idx][row_idx].extend([None for ins_idx in range(cols - cur_cols)])
                    self.attrs[layer_idx][row_idx].extend([None for ins_idx in range(cols - cur_cols)])

        self._invalidate()

        # Update config
        self.config["dims"] = (cols, rows)

        return True


class DrawCanvas(TextGrid):
    """ Provide text grid with tool functionality """
    TOOLS = [["Paint", "Erase", "Draw", "Text", "Select", "Stamp"], ["Image", "Layer"]]
    TOOL_DEFAULTS = {"Paint": {"size": 1}, "Erase": {"size": 1}, "Draw": {"size": 1}, "Text": {"active": False}, "Select": {"active": False}, "Stamp": {"original": True}}
    TOOL_RESET = {"Text": {"active": False}, "Select": {"active": False}}

    @classmethod
    def attr_ansi(cls, attr):
        """ Convert attributes to ANSI escape codes """
        # Foreground
        mod = 30 if attr[0] < 8 else 82
        fore = "{}".format(attr[0] + mod)

        # Background
        mod = 40 if attr[1] < 8 else 92
        back = "{}".format(attr[1] + mod)

        return bytes("\x1B[{}m\x1B[{}m".format(fore, back).encode("utf-8"))

    def __init__(self, wm, config={}):
        super().__init__(wm, config=config)
        self.active_attrs = [0, 0, 0]
        self.pointer_tool = self.TOOLS[0][0]
        self.options_tool = self.TOOLS[0][0]
        self.active_symbol = b""
        self.option_vals = {tool: dict(DrawCanvas.TOOL_DEFAULTS[tool]) for tool in DrawCanvas.TOOL_DEFAULTS}

    def selectable(self):
        return True

    def keypress(self, size, key):
        """ Handle canvas key events """
        if self.option_vals["Text"]["active"]:
            # Intercept keys for text mode
            self.text_plot(key.encode("utf-8"))
        else:
            return key

    def grid_event(self, size, event, button, col, row, focus):
        """ Handle drawing mouse events """
        options = self.option_vals.setdefault(self.pointer_tool, {})

        # Mouse release
        if int(button) == 0:
            if self.pointer_tool in ["Paint", "Erase", "Draw", "Stamp"]:
                # Turn off hint cursor after releasing button
                self.clear(layer=-1)

            if self.pointer_tool == "Select":
                options["active"] = False

        # Left button
        if int(button) == 1:
            if self.pointer_tool == "Paint":
                self.plot(col, row, b' ', (self.active_attrs[0], self.active_attrs[0], 0), size=options["size"])

            if self.pointer_tool == "Erase":
                self.plot(col, row, None, None, size=options["size"])

            if self.pointer_tool == "Draw":
                self.plot(col, row, self.active_symbol, (self.active_attrs[0], self.active_attrs[1], 0), size=options["size"])

            if self.pointer_tool == "Text":
                options["active"] = True
                options["col"] = col
                options["row"] = row
                options["col_start"] = col
                options["row_start"] = row

                self.clear(layer=-1)
                self.plot(col, row, None, None, layer=-1, size=1, inverse=True)

            if self.pointer_tool == "Select":
                self.select_area(col, row, reselect=options["active"])
                self.plot_select()
                options["active"] = True

            if self.pointer_tool == "Stamp":
                self.plot_stamp(col, row, self.layer)

        # Right button
        if int(button) == 3:
            if self.pointer_tool in ["Paint", "Erase", "Draw"]:
                # Display hint cursor
                self.clear(layer=-1)
                self.plot(col, row, None, None, layer=-1, size=self.option_vals[self.pointer_tool]["size"], inverse=True)

            if self.pointer_tool == "Select":
                # Reselect end bounds
                self.select_area(col, row)
                self.plot_select()

            if self.pointer_tool == "Stamp":
                # Stamp hint
                self.clear(layer=-1)
                self.plot_stamp(col, row)

    def plot(self, col, row, char, attrs, layer=None, size=1, inverse=False):
        """ Extended plotting (size control, inverse) """
        start_col = max(0, col - math.ceil(size / 2) + 1)
        start_row = max(0, row - math.ceil(size / 2) + 1)
        dims = self.get_dims()

        for col_idx in range(start_col, start_col + size):
            for row_idx in range(start_row, start_row + size):
                # Check bounds
                if row_idx >= dims[2] or col_idx >= dims[1]:
                    break

                cur_attr = attrs
                if inverse:
                    old_attr = self.attrs[self.layer][row_idx][col_idx]
                    inv_attr = list(old_attr if old_attr is not None else (0, 0, 0))
                    inv_attr[0] = (inv_attr[0] + 7) % 16
                    inv_attr[1] = (inv_attr[1] + 7) % 16
                    cur_attr = tuple(inv_attr)

                super().plot(col_idx, row_idx, char, cur_attr, layer)

    def text_plot(self, char):
        """ Plot character from text tool """
        options = self.option_vals["Text"]

        if len(char) > 1:
            if char == b"backspace":
                # Calculate if a non-background/transparent char underneath cursor
                existing_char = False
                if self.content[self.layer][options["row"]][options["col"]] is not None:
                    existing_char = True

                if options["row"] == options["row_start"] and options["col"] == options["col_start"]:
                    # Starting cursor location, do nothing
                    return

                if options["col"] >= len(self.content[0][0]) - 1 and existing_char:
                    # Cursor at end of line with something underneath, delete current character and do not move cursor
                    pass
                elif options["col"] > options["col_start"]:
                    # Cursor at non-start of line, delete previous character and move cursor back
                    options["col"] -= 1
                else:
                    # Cursor at start of line, move up a line
                    options["col"] = options["col_end"]
                    options["row"] -= 1

                # Set to transparent
                self.plot(options["col"], options["row"], None, None, size=1)

            elif char == b"enter":
                options["col_end"] = options["col"]
                options["col"] = options["col_start"]
                options["row"] += 1
            elif char == b"esc":
                options["active"] = False
            else:
                # Unrecognized character
                return
        else:
            self.plot(options["col"], options["row"], char, list(self.active_attrs))
            options["col"] += 1

        # Check bounds
        if options["col"] >= len(self.content[0][0]):
            options["col"] = len(self.content[0][0]) - 1
        if options["row"] >= len(self.content[0]):
            options["row"] = len(self.content[0]) - 1

        # Keep cursor on if active
        self.clear(layer=-1)
        if options["active"]:
            self.plot(options["col"], options["row"], None, None, layer=-1, inverse=True)

    def plot_select(self):
        """ Plot selection rectangle """
        options = self.option_vals["Select"]

        self.clear(layer=-1)
        row_step = 1 if options["row_start"] < options["row_end"] else -1
        for row_idx in range(options["row_start"], options["row_end"] + row_step, row_step):
            self.plot(options["col_start"], row_idx, None, None, layer=-1, size=1, inverse=True)
            self.plot(options["col_end"], row_idx, None, None, layer=-1, size=1, inverse=True)

        col_step = 1 if options["col_start"] < options["col_end"] else -1
        for col_idx in range(options["col_start"], options["col_end"] + col_step, col_step):
            self.plot(col_idx, options["row_start"], None, None, layer=-1, size=1, inverse=True)
            self.plot(col_idx, options["row_end"], None, None, layer=-1, size=1, inverse=True)

    def plot_stamp(self, col, row, layer=-1):
        """ Plot clip buffer """
        options_select = self.option_vals["Select"]
        options_stamp = self.option_vals["Stamp"]

        if "clip_content" not in options_select:
            return

        for row_idx in range(len(options_select["clip_content"])):
            for col_idx in range(len(options_select["clip_content"][0])):
                char = options_select["clip_content"][row_idx][col_idx]
                attr = options_select["clip_attrs"][row_idx][col_idx]

                if not options_stamp["original"]:
                    if char == b" " and attr[0] == attr[1]:
                        attr = (self.active_attrs[0], self.active_attrs[0], 0)
                    else:
                        attr = (self.active_attrs[0], self.active_attrs[1], 0)

                if char is not None:
                    # Only stamp non-transparent characters
                    self.plot(col + col_idx, row + row_idx, char, attr, layer=layer)

    def select_area(self, col, row, reselect=True):
        """ Select area of canvas """
        options = self.option_vals["Select"]

        if ("col_start" not in options) or not reselect:
            options["col_start"] = col
            options["row_start"] = row

        options["col_end"] = col
        options["row_end"] = row

        # Check bounds
        options["col_start"] = min(options["col_start"], len(self.content[self.layer][0]) - 1)
        options["col_end"] = min(options["col_end"], len(self.content[self.layer][0]) - 1)
        options["row_start"] = min(options["row_start"], len(self.content[self.layer]) - 1)
        options["row_end"] = min(options["row_end"], len(self.content[self.layer]) - 1)

    def copy_area(self, copy, clear):
        """ Cut/Copy/Clear selected area to clipboard """
        options = self.option_vals["Select"]

        # Check if no area selected
        if "row_start" not in options:
            return

        clip_content = []
        clip_attrs = []

        row_bounds = sorted([options["row_start"], options["row_end"]])
        col_bounds = sorted([options["col_start"], options["col_end"]])

        for row_idx in range(row_bounds[0], row_bounds[1] + 1):
            clip_content.append([])
            clip_attrs.append([])

            for col_idx in range(col_bounds[0], col_bounds[1] + 1):
                clip_content[-1].append(self.content[self.layer][row_idx][col_idx])
                clip_attrs[-1].append(self.attrs[self.layer][row_idx][col_idx])

                # Remove if clear mode
                if clear:
                    self.content[self.layer][row_idx][col_idx] = None
                    self.attrs[self.layer][row_idx][col_idx] = None

        if copy:
            options["clip_content"] = clip_content
            options["clip_attrs"] = clip_attrs

        if clear:
            self._invalidate()

    def save(self, path):
        """ Save all non-cursor layers to file """
        dims = self.get_dims()
        full_path = os.path.expanduser(path)

        with open(full_path, "wb") as handle:
            # Write layer count
            handle.write(struct.pack("I", dims[0] - 1))

            for layer_idx in range(dims[0] - 1):
                # Write row count
                handle.write(struct.pack("I", dims[2]))

                for row_idx in range(dims[2]):
                    # Write column count
                    handle.write(struct.pack("I", dims[1]))

                    for col_idx in range(dims[1]):
                        char = self.content[layer_idx][row_idx][col_idx]
                        char = b"\0\0\0\0" if char is None else char.ljust(4, b"\0")
                        attr = self.attrs[layer_idx][row_idx][col_idx]
                        attr = (255, 255) if attr is None else attr

                        # Write unicode character as fixed 4-byte value
                        handle.write(char)

                        # Write attribute as fixed 2-byte value (FG,BG)
                        handle.write(attr[0].to_bytes(1, "little"))
                        handle.write(attr[1].to_bytes(1, "little"))

    def load(self, path):
        """ Load non-cursor layers from a file """
        full_path = os.path.expanduser(path)
        self.reset()

        with open(full_path, "rb") as handle:
            # Read layer count
            layers = handle.read(4)
            layers = struct.unpack("I", layers)[0]

            for layer_idx in range(layers):
                # Read row count
                rows = handle.read(4)
                rows = struct.unpack("I", rows)[0]

                # Add layers above 0
                if layer_idx > 0:
                    self.add_layer(layer_idx)

                self.content[layer_idx].clear()
                self.attrs[layer_idx].clear()

                for row_idx in range(rows):
                    self.content[layer_idx].append([])
                    self.attrs[layer_idx].append([])

                    # Read column count
                    cols = handle.read(4)
                    cols = struct.unpack("I", cols)[0]

                    for col_idx in range(cols):
                        # Read unicode character as fixed 4-byte value
                        char = handle.read(4).rstrip(b"\0")
                        char = None if char == b"" else char
                        self.content[layer_idx][row_idx].append(char)

                        # Read attribute as fixed 2-byte value (FG,BG)
                        fore = int.from_bytes(handle.read(1), "little")
                        back = int.from_bytes(handle.read(1), "little")
                        attr = None if (fore == 255 and back == 255) else (fore, back, 0)
                        self.attrs[layer_idx][row_idx].append(attr)

    def export_text(self, path, color):
        """ Export text version of flattened image """
        full_path = os.path.expanduser(path)
        flat_text, flat_attrs = self.get_flattened()

        with open(full_path, "wb") as handle:
            for row_idx in range(len(flat_text)):
                for col_idx in range(len(flat_text[row_idx])):
                    if color:
                        code = DrawCanvas.attr_ansi(flat_attrs[row_idx][col_idx])
                        handle.write(code)

                    char = flat_text[row_idx][col_idx]
                    handle.write(char)

                handle.write(b"\n")

            # End file with attribute reset
            handle.write(b"\x1B[0m")

    def import_image(self, path):
        """ Import and convert supported image """
        full_path = os.path.expanduser(path)
        dims = self.get_dims()
        self.reset()

        with Image.open(full_path, "r") as handle:
            pixels = handle.getdata()

            # Calculate pixels per character
            ppc_width = handle.width / dims[1]
            ppc_height = handle.height / dims[2]

            # Generate chunk coordinates
            chunks_width = [int(idx * ppc_width) for idx in range(int(handle.width / ppc_width))]
            chunks_height = [int(idx * ppc_height) for idx in range(int(handle.height / ppc_height))]
            chunks_width.append(handle.width)
            chunks_height.append(handle.height)

            for chunk_y in range(len(chunks_height) - 1):
                for chunk_x in range(len(chunks_width) - 1):
                    chunk_pixels = []

                    # Calculate mean chunk color
                    for idx_y in range(chunks_height[chunk_y], chunks_height[chunk_y + 1]):
                        for idx_x in range(chunks_width[chunk_x], chunks_width[chunk_x + 1]):
                            chunk_pixels.append(pixels[idx_y * handle.width + idx_x])

                    chunk_pixels = zip(*chunk_pixels)
                    chunk_color = [int(sum(channel) / len(channel)) for channel in chunk_pixels]
                    NetPaint.inst.log(chunk_color)
                    # Calculate color distances
                    distances = []
                    for idx, color in enumerate(NetPaint.inst.RGB_LOOKUP):
                        distance = math.sqrt((chunk_color[0] - color[0])**2 + (chunk_color[1] - color[1])**2 + (chunk_color[2] - color[2])**2)
                        distances.append((distance, idx))

                    # Get primary and secondary, calculate ratio, choose character
                    distances = sorted(distances)
                    ratio = distances[0][0] / distances[1][0]
                    if ratio > 0.66:
                        char = bytes(chr(9618), "utf8")
                    elif ratio > 0.33:
                        char = bytes(chr(9617), "utf8")
                    else:
                        char = b" "
                    
                    # Draw character
                    self.content[0][chunk_y][chunk_x] = char
                    self.attrs[0][chunk_y][chunk_x] = (distances[1][1], distances[0][1], 0)

    def select_tool(self, tool):
        """ Setup selected tool"""
        self.options_tool = tool
        if tool in DrawCanvas.TOOLS[0]:
            self.pointer_tool = tool

        # Reset all tools
        for tool in DrawCanvas.TOOL_RESET:
            for option, val in DrawCanvas.TOOL_RESET[tool].items():
                self.option_vals[tool][option] = val


class SymbolSelect(TextGrid):
    """ Provide text grid with symbol selection functionality """
    def __init__(self, wm, symbols):
        super().__init__(wm, config={"scroll_h": False})
        self.active_row = 0
        self.active_col = 0

        pos = 0
        for sym_set in symbols:
            for sym_idx in range(*sym_set):
                self.plot(pos % 20, pos // 20, bytes(chr(sym_idx), "utf8"), [15, 0, 0])
                pos += 1

        self.select_symbol(0, 0)

    def grid_event(self, size, event, button, col, row, focus):
        """ Process mouse events """
        if int(button) == 1:
            # Check symbol was clicked
            if row < len(self.content[0]) and col < len(self.content[0][0]):
                self.select_symbol(col, row)

    def select_symbol(self, col, row):
        """ Select symbol from grid """
        self.attrs[0][self.active_row][self.active_col] = [15, 0, 0]
        self.attrs[0][row][col] = [0, 6, 0]
        self.active_row, self.active_col = row, col
        self._invalidate()

        self.wm.get("canvas", True).active_symbol = self.content[0][row][col]


class TextArea(TextGrid):
    """ Provide text grid with printed text functionality """
    def __init__(self, wm, text, attr, width):
        super().__init__(wm, config={"scroll_h": False})
        self.width = width
        self.set_text(text, attr)

    def set_text(self, text, attr):
        """ Plot text on grid """
        # Set background color
        self.config["back_attrs"] = attr

        row = 0
        col = 0

        for char_idx in range(len(text)):
            if text[char_idx] == "\n" or col >= self.width or self.wrap_word(col, text[char_idx:]):
                col = 0
                row += 1

            if text[char_idx] != "\n":
                self.plot(col, row, bytes(text[char_idx], "utf8"), attr)
                col += 1

    def wrap_word(self, col, remaining):
        """ Detect if word needs to be wrapped """
        next_space = remaining.find(" ")

        if next_space == -1:
            return False
        else:
            return col + next_space >= self.width


class ColorSelect(urwid.SolidFill):
    """ Provide foreground/background color selection functionality """
    class ColorButton(urwid.SolidFill):
        """ Provide color selection button functionality """
        def __init__(self, parent, color):
            super().__init__()
            self.parent = parent
            self.color = color

        def mouse_event(self, size, event, button, col, row, focus):
            """ Process mouse events """
            self.parent.set_color(self.color)
            self.parent.wm.deactivate_overlay()

    def __init__(self, wm, canvas, color, select):
        super().__init__(b" ")
        self.wm = wm
        self.canvas = canvas
        self.color = color
        self.select = select
        self.palette_id = "palette_{}".format(self.select)
        self.wm.get(self.canvas, True).active_attrs[self.select] = self.color

        self.setup_widgets()

    def setup_widgets(self):
        """ Setup color button widgets """
        for color in range(16):
            color_id = "{0}-{0}".format(self.wm.get_color_name(color))
            button_name = "{}_button_{}".format(self.palette_id, color)
            self.wm.register(button_name, urwid.BoxAdapter(self.ColorButton(self, color), 2), color_id)
            self.wm.group_add("{}_buttons".format(self.palette_id), self.wm[button_name])

        self.wm["{}_grid".format(self.palette_id)] = urwid.GridFlow(self.wm.group_get("{}_buttons".format(self.palette_id)), 4, 1, 1, "center")
        self.wm["{}_filler".format(self.palette_id)] = urwid.Filler(self.wm["{}_grid".format(self.palette_id)], height="pack")
        self.wm["{}_box".format(self.palette_id)] = urwid.LineBox(self.wm["{}_filler".format(self.palette_id)], "Select Color", title_align="left")
        self.wm["{}_overlay".format(self.palette_id)] = urwid.Overlay(self.wm["{}_box".format(self.palette_id)], self.wm[self.canvas], align="center", width=44, valign="middle", height=9)

    def set_color(self, color):
        """ Set active color """
        self.color = color
        self.wm.get(self.canvas, True).active_attrs[self.select] = self.color
        self._invalidate()

    def mouse_event(self, size, event, button, col, row, focus):
        """ Process mouse events """
        if button == 1:
            self.wm.activate_overlay("{}_overlay".format(self.palette_id))

    def render(self, size, focus=False):
        """ Render color buttons """
        canvas = urwid.CompositeCanvas(super().render(size, focus))
        canvas.fill_attr("{0}-{0}".format(self.wm.get_color_name(self.color)))
        return canvas


class DialogBox(urwid.Overlay):
    """ Provide customizable dialog box functionality """
    FIELD_DEFAULTS = {"width": 60, "height": 14, "msg": "", "buttons": None, "edit": None, "scroll": None, "align": "left", "space": 4}

    def __init__(self, name, wm, canvas, handler, config):
        self.name = name
        self.wm = wm
        self.canvas = canvas
        self.handler = handler
        self.config = dict(self.FIELD_DEFAULTS)
        self.config.update(config)
        self.button_width = 15
        self.margin = 2

        # Setup dialog sub-widgets
        self.setup_widgets()
        super().__init__(self.wm["{}_box".format(self.name)], self.wm[self.canvas], align="center", width=self.config["width"], valign="middle", height=self.config["height"])

    def setup_widgets(self):
        """ Create dialog widgets """
        # Register general widgets
        if self.config["scroll"] is None:
            self.wm["{}_text".format(self.name)] = urwid.Text(self.config["msg"], align=self.config["align"])
        else:
            self.wm["{}_text".format(self.name)] = urwid.BoxAdapter(TextArea(self.wm, self.config["msg"], (14, 4, 0), self.config["width"] - 10), self.config["scroll"])

        self.wm["{}_solid".format(self.name)] = urwid.BoxAdapter(urwid.SolidFill(" "), 1)

        if self.config["edit"] is not None:
            self.wm.register("{}_edit".format(self.name), urwid.Edit(self.config["edit"]), "dialog-edit")
        else:
            self.wm.register("{}_edit".format(self.name), self.wm["{}_solid".format(self.name)])

        # Register button widgets
        buttons = []
        gap_total = self.config["width"] - self.button_width * len(self.config["buttons"])
        gap_width = int(gap_total / (len(self.config["buttons"]) + 1))

        for button in self.config["buttons"]:
            label = "{}{}".format(int((self.button_width - len(button) - 5) / 2) * " ", button)
            self.wm.register("{}_button_{}".format(self.name, button), urwid.Button(label, on_press=self.handler, user_data=None, ), "dialog-button")
            buttons.append(self.wm["{}_button_{}".format(self.name, button)])
            buttons.append((gap_width, self.wm["{}_solid".format(self.name)]))

        self.wm["{}_mouse_cols".format(self.name)] = urwid.Columns(buttons[:-1])
        self.wm["{}_mouse_padding".format(self.name)] = urwid.Padding(self.wm["{}_mouse_cols".format(self.name)], align="center", left=gap_width - self.margin, right=gap_width - self.margin)
        group = "{}_input".format(self.name)

        add_space = lambda num: [self.wm.group_add(group, self.wm["{}_solid".format(self.name)]) for x in range(num)]
        self.wm.group_add(group, self.wm["{}_text".format(self.name)])
        add_space(1)
        self.wm.group_add(group, self.wm["{}_edit".format(self.name)])
        add_space(self.config["space"])
        self.wm.group_add(group, self.wm["{}_mouse_padding".format(self.name)])

        # Register layout widgets
        self.wm["{}_pile".format(self.name)] = urwid.Pile(self.wm.group_get("{}_input".format(self.name)))
        self.wm["{}_filler".format(self.name)] = urwid.Filler(self.wm["{}_pile".format(self.name)])
        self.wm["{}_padding".format(self.name)] = urwid.Padding(self.wm["{}_filler".format(self.name)], align="center", left=2, right=2)
        self.wm["{}_box".format(self.name)] = urwid.LineBox(self.wm["{}_padding".format(self.name)], self.config["title"], title_align="left")

    def reset(self):
        """ Reset all field values """
        if self.config["edit"] is not None:
            self.wm.get("{}_edit".format(self.name), True).set_edit_text(self.config["edit"])


class NetPaint:
    """ Provides general NetPaint functionality (singleton) """
    VERSION = "1.1.2"

    ENCODING = "utf8"

    PALETTE = [("toolbox", "white", "dark gray"),
               ("menu", "black", "light gray"),
               ("menu-selected", "light gray", "black"),
               ("dialog", "light cyan", "dark blue"),
               ("dialog-edit", "black", "light cyan"),
               ("dialog-button", "black", "white"),
               ("option-edit", "black", "light gray"),
               ("status", "light cyan", "dark blue"),
               ("canvas", "yellow", "black")]

    RGB_LOOKUP = [(0, 0, 0), (0x7F, 0, 0), (0, 0x7F, 0), (0x7F, 0x7F, 0),
                  (0, 0, 0x7F), (0x7F, 0, 0x7F), (0, 0x7F, 0x7F), (0xC0, 0xC0, 0xC0),
                  (0x7F, 0x7F, 0x7F), (0xFF, 0, 0), (0, 0xFF, 0), (0xFF, 0xFF, 0),
                  (0, 0, 0xFF), (0xFF, 0, 0xFF), (0, 0xFF, 0xFF), (0xFF, 0xFF, 0xFF)]

    WELCOME_MSG = ("Welcome to NetPaint, the text-based drawing program! "
                   "This program is inspired by creative ASCII art "
                   "and many fond memories of dialup BBSes.\n\n"
                   "If you have any issues or suggestions "
                   "regarding this program, please visit github.com/SyntheticDreams, or "
                   "find me on Twitter: @ToniWestbrook.  Thanks!\n\n"
                   "Note: Please ensure your terminal has mouse support!")

    HELP_MSG = ("NetPaint provides the following tools, selectable using the toolbox or pressing the associated hotkey:\n\n"
                "Paint (p)\n----------\n"
                "L: Paint using solid rectangle of the foreground color.  The 'Brush Size' option controls the size of the rectangle.\n\n"
                "R: Display hint cursor.\n\n\n"
                "Erase (e)\n----------\n"
                "L: Erase using solid rectangle (sets to transparent).  The 'Brush Size' option controls the size of the rectangle.\n\n"
                "R: Display hint cursor.\n\n\n"
                "Draw (d)\n----------\n"
                "L: Draw using rectangle of the selected symbol, drawn using foreground and background colors.  The 'Brush Size' option controls the size of the rectangle.\n\n"
                "R: Display hint cursor.\n\n\n"
                "Text (t)\n----------\n"
                "L: Begin writing text using foreground and background colors. Press enter to advance to next line, and escape to exit text mode.\n\n\n"
                "Select (s)\n----------\n"
                "L: Select a new rectangular region for cut/copy/clear (Edit menu).\n\n"
                "R: Extend selected rectangular region.\n\n\n"
                "Stamp (m)\n----------\n"
                "L: Paste the cut/copied region using the original or active colors.\n\n"
                "R: Display the stamp hint cursor.\n\n\n"
                "Image (i)\n----------\n"
                "View/edit image dimensions.  Click apply to update or reset to revert.\n\n\n"
                "Layer (l)\n----------\n"
                "View/edit layer information.  All tools will update the currently active layer.  All layers except 0 may be set as visible or hidden.  "
                "Flattening the image will combine all layers into layer 0 (applied in order).\n\n\n"
                "Other hotkeys\n----------\n"
                "Cut (c)\n"
                "Copy (y)\n"
                "Clear (del)\n"
                "Select all (a)\n"
                "Quit (q)\n"
                "")

    SYMBOLS = [(33, 127), (9472, 9600), (9617, 9620)]

    inst = None

    @classmethod
    def setup_runtime_constants(cls):
        """ Setup runtime constants requiring NetPaint instance """
        cls.GLOBAL_KEYS = {"q": partial(cls.inst.quit),
                           "p": partial(cls.inst.select_tool, "Paint", widget=True),
                           "e": partial(cls.inst.select_tool, "Erase", widget=True),
                           "d": partial(cls.inst.select_tool, "Draw", widget=True),
                           "t": partial(cls.inst.select_tool, "Text", widget=True),
                           "s": partial(cls.inst.select_tool, "Select", widget=True),
                           "m": partial(cls.inst.select_tool, "Stamp", widget=True),
                           "i": partial(cls.inst.select_tool, "Image", widget=True),
                           "l": partial(cls.inst.select_tool, "Layer", widget=True),
                           "a": partial(cls.inst.select_all),
                           "c": partial(cls.inst.canvas.copy_area, copy=True, clear=True),
                           "y": partial(cls.inst.canvas.copy_area, copy=True, clear=False),
                           "delete": partial(cls.inst.canvas.copy_area, copy=False, clear=True),
                           "esc": partial(cls.inst.reset_menu)}

    @classmethod
    def instance(cls):
        if cls.inst is None:
            cls.inst = cls()

        return cls.inst

    def setup(self, args):
        """ Setup all widgets, program options, and urwid functionality """
        # Initialize urwid
        urwid.set_encoding(self.ENCODING)

        # Create widget manager, connect to netpaint, and setup widgets
        self.args = args
        self.wm = WidgetManager(self.PALETTE, self.ENCODING)
        self.setup_widgets()
        self.wm.register_base("body", "canvas")
        self.wm.activate_overlay("dialog_welcome_overlay")

        # Finalize setup
        self.setup_runtime_constants()
        self.main_loop = urwid.MainLoop(self.wm["screen"], self.PALETTE, unhandled_input=self.key_handler, input_filter=self.filter_handler)
        self.resize_screen()
        self.main_loop.set_alarm_in(1, self.time_handler)

        # Load file if specified
        if args.image:
            self.canvas.load(args.image)

    def start(self):
        """ Start main loop """
        self.main_loop.run()

    def log(self, msg):
        """ Log debug message """
        with open("debug.log", "a") as handle:
            print(msg, file=handle)

    def log_status(self, msg, error=False):
        """ Update status message """
        prepend = "Error:" if error else "Info:"
        self.wm.get("status", True).set_text("{} {}".format(prepend, msg))

    def quit(self):
        """ Exit program """
        raise urwid.ExitMainLoop()

    def key_handler(self, key):
        """ Process global keyboard events """
        if key in self.GLOBAL_KEYS:
            self.GLOBAL_KEYS[key]()

    def filter_handler(self, keys, raw):
        """ Process resize events """
        if "window resize" in keys:
            self.resize_screen()

        return keys

    def menu_handler(self, button, command=None):
        """ Menu button handler """
        wm = self.wm
        # Note existing overlay, if any
        current_overlay = wm.get_overlay()

        # Reset menu
        self.reset_menu()

        # Exit if deselecting menu button
        if command and wm.get(command, True) == current_overlay:
            return

        # Dropdown activation
        if button == wm.get("menu_file", True):
            wm.activate_overlay(command)
            wm.update_attr("menu_file", "menu-selected")

        if button == wm.get("menu_edit", True):
            wm.activate_overlay(command)
            wm.update_attr("menu_edit", "menu-selected")

        if button == wm.get("menu_view", True):
            wm.activate_overlay(command)
            wm.update_attr("menu_view", "menu-selected")

        if button == wm.get("menu_help", True):
            wm.activate_overlay(command)
            wm.update_attr("menu_help", "menu-selected")

        # Item commands
        if button == wm.get("menu_file_new", True):
            self.canvas.reset()
            self.canvas.scroll(0, 0, rel=False)

        if button == wm.get("menu_file_open", True):
            wm.activate_overlay("dialog_open_overlay")

        if button == wm.get("menu_file_save", True):
            wm.activate_overlay("dialog_save_overlay")

        if button == wm.get("menu_file_imimg", True):
            wm.activate_overlay("dialog_importimg_overlay")

        if button == wm.get("menu_file_exbw", True):
            wm.activate_overlay("dialog_exportbw_overlay")

        if button == wm.get("menu_file_exclr", True):
            wm.activate_overlay("dialog_exportcolor_overlay")

        if button == wm.get("menu_file_exit", True):
            raise urwid.ExitMainLoop()

        if button == wm.get("menu_edit_cut", True):
            self.canvas.copy_area(copy=True, clear=True)

        if button == wm.get("menu_edit_copy", True):
            self.canvas.copy_area(copy=True, clear=False)

        if button == wm.get("menu_edit_clear", True):
            self.canvas.copy_area(copy=False, clear=True)

        if button == wm.get("menu_edit_select", True):
            self.select_all()

        if button == wm.get("menu_view_show", True):
            self.show_tools(True)

        if button == wm.get("menu_view_hide", True):
            self.show_tools(False)

        if button == wm.get("menu_help_help", True):
            wm.activate_overlay("dialog_help_overlay")

        if button == wm.get("menu_help_about", True):
            wm.activate_overlay("dialog_welcome_overlay")

    def dialog_handler(self, button, command=None):
        """ Dialog button handler """
        wm = self.wm

        if button == wm.get("dialog_open_button_Open", True):
            try:
                self.canvas.load(wm.get("dialog_open_edit", True).text)
                self.select_layer()
                self.log_status("File opened successfully")
            except Exception as e:
                self.log_status(e, error=True)

        if button == wm.get("dialog_save_button_Save", True):
            try:
                self.canvas.save(wm.get("dialog_save_edit", True).text)
                self.log_status("File saved successfully")
            except Exception as e:
                self.log_status(e, error=True)

        if button == wm.get("dialog_importimg_button_Open", True):
            try:
                self.canvas.import_image(wm.get("dialog_importimg_edit", True).text)
                self.log_status("File imported successfully")
            except Exception as e:
                self.log_status(e, error=True)

        if button == wm.get("dialog_exportbw_button_Save", True):
            try:
                self.canvas.export_text(wm.get("dialog_exportbw_edit", True).text, color=False)
                self.log_status("File exported successfully")
            except Exception as e:
                self.log_status(e, error=True)

        if button == wm.get("dialog_exportcolor_button_Save", True):
            try:
                self.canvas.export_text(wm.get("dialog_exportcolor_edit", True).text, color=True)
                self.log_status("File exported successfully")
            except Exception as e:
                self.log_status(e, error=True)

        # Buttons close dialog by default
        wm.deactivate_overlay()

    def tool_handler(self, button, command=None):
        """ Tool change related handler """
        # Process button presses only
        if not command:
            return

        # Check change to active tool
        for tool_group in DrawCanvas.TOOLS:
            for tool in tool_group:
                if button == self.wm.get("tool_{}_button".format(tool), True):
                    self.select_tool(tool, widget=False)

    def option_handler(self, widget, command=None):
        """ Option change handler """
        wm = self.wm

        # Paint options
        if widget == wm.get("options_Paint_1_size", True):
            size = int(command) if command.isdigit() else 1
            self.canvas.option_vals["Paint"]["size"] = max(size, 1)

        # Erase options
        if widget == wm.get("options_Erase_1_size", True):
            size = int(command) if command.isdigit() else 1
            self.canvas.option_vals["Erase"]["size"] = max(size, 1)

        # Draw options
        if widget == wm.get("options_Draw_2_size", True):
            size = int(command) if command.isdigit() else 1
            self.canvas.option_vals["Draw"]["size"] = max(size, 1)

        # Image options
        reset_dims = False

        if widget == wm.get("options_Image_4_reset", True):
            reset_dims = True

        if widget == wm.get("options_Image_5_apply", True):
            width = wm.get("options_Image_1_width", True).edit_text
            width = int(width) if width.isdigit() else 0

            height = wm.get("options_Image_2_height", True).edit_text
            height = int(height) if height.isdigit() else 0

            self.canvas.scroll(0, 0, rel=False)
            reset_dims = not self.canvas.set_dims(width, height)

        if reset_dims:
            wm.get("options_Image_1_width", True).edit_text = str(self.canvas.get_dims()[1])
            wm.get("options_Image_2_height", True).edit_text = str(self.canvas.get_dims()[2])

        # Stamp options
        if widget == wm.get("options_Stamp_1_original", True) and command:
            self.canvas.option_vals["Stamp"]["original"] = True

        if widget == wm.get("options_Stamp_1_current", True) and command:
            self.canvas.option_vals["Stamp"]["original"] = False

        # Layer options
        layer_change = False

        if widget == wm.get("options_Layer_1_current", True):
            layer = int(command) if command.isdigit() else 0
            layer_change = self.canvas.select_layer(layer)

        if widget == wm.get("options_Layer_2_visible", True) and command:
            self.canvas.show_layer(self.canvas.layer, True)
            self.canvas._invalidate()

        if widget == wm.get("options_Layer_2_hidden", True) and command:
            self.canvas.show_layer(self.canvas.layer, False)
            self.canvas._invalidate()

        if widget == wm.get("options_Layer_3_add", True):
            if self.canvas.add_layer(self.canvas.layer + 1):
                layer_change = self.canvas.select_layer(self.canvas.layer + 1)

        if widget == wm.get("options_Layer_4_del", True):
            if self.canvas.del_layer(self.canvas.layer):
                layer_change = self.canvas.select_layer(self.canvas.layer - 1)

        if widget == wm.get("options_Layer_5_flat", True):
            self.canvas.content[0], self.canvas.attrs[0] = self.canvas.get_flattened()
            while self.canvas.del_layer(len(self.canvas.content) - 2):
                layer_change = self.canvas.select_layer(0)

        if layer_change:
            self.select_layer(widget != wm.get("options_Layer_1_current", True))

    def time_handler(self, loop, user_data):
        """ Update status bar clock """
        self.wm.get("time", True).set_text("{}".format(time.strftime(" | %H:%M:%S | %d-%b-%y ")))
        loop.set_alarm_in(0.1, self.time_handler)

    def select_tool(self, tool, widget=True):
        """ Set active tool and associated options """
        self.canvas.clear(layer=-1)

        # Notify canvas of tool change
        self.canvas.select_tool(tool)

        # Set tool button if requested
        if widget:
            self.wm.get("tool_{}_button".format(tool), True).set_state(True, True)

        # Update options for options tool
        self.wm.group_get("toolbox").clear()
        self.wm.group_add("toolbox", (self.wm["tool_box"], self.wm.get("toolbox", True).options("pack")))
        self.wm.group_add("toolbox", (self.wm["empty_box"], self.wm.get("toolbox", True).options("given", 2)))

        for option in self.wm.group_get("options_{}".format(tool)):
            self.wm.group_add("toolbox", (option, self.wm.get("toolbox", True).options("pack")))

        self.wm.group_add("toolbox", (self.wm["empty_box"], self.wm.get("toolbox", True).options("weight", 1)))
        self.wm.group_add("toolbox", (self.wm["color_box"], self.wm.get("toolbox", True).options("pack")))
        self.wm.get("toolbox", True).contents = [option for option in self.wm.group_get("toolbox")]

    def select_layer(self, widget=True):
        """ Select current layer in options """
        if widget:
            # Only update text widget if requested (avoid recursive loop if triggered from text change)
            self.wm.get("options_Layer_1_current", True).edit_text = str(self.canvas.layer)

        self.wm.get("options_Layer_2_visible", True).set_state(self.canvas.visibility[self.canvas.layer], False)
        self.wm.get("options_Layer_2_hidden", True).set_state(not self.canvas.visibility[self.canvas.layer], False)
        self.wm.get("options_Layer_1_box", True).title_widget.set_text(" Current [0-{}] ".format(len(self.canvas.content) - 2))

    def select_all(self):
        """ Select entire canvas """
        self.select_tool("Select")
        self.canvas.select_area(0, 0, reselect=False)
        self.canvas.select_area(self.canvas.get_dims()[1], self.canvas.get_dims()[2], reselect=True)
        self.canvas.plot_select()

    def show_tools(self, enabled):
        """ Enable or disable tool sidebar """
        body = self.wm.get("body", True)

        if enabled:
            body.contents = [(self.wm["canvas"], body.options()), (self.wm["toolbox"], body.options("given", 23))]
        else:
            body.contents = [(self.wm["canvas"], body.options())]

    def resize_screen(self):
        """ Process screen resizing """
        size = self.main_loop.screen.get_cols_rows()

        if size[0] < 4 or size[1] < 4:
            body = (self.wm.get("empty_box", True), self.wm.get("screen", True).options())
        else:
            body = (self.wm.get("body"), self.wm.get("screen", True).options())

        self.wm.get("screen", True).contents["body"] = body

    def reset_menu(self):
        """ Reset menu bar """
        self.wm.deactivate_overlay()
        self.wm.update_attr("menu_file", "menu")
        self.wm.update_attr("menu_edit", "menu")
        self.wm.update_attr("menu_view", "menu")
        self.wm.update_attr("menu_help", "menu")

    def setup_widgets(self):
        """ Create all widgets """
        wm = self.wm

        # Alter default urwid behavior
        urwid.Button.button_left = urwid.Text("")
        urwid.Button.button_right = urwid.Text("")

        wm["canvas"] = DrawCanvas(wm, {"dims": (self.args.d[0], self.args.d[1])})
        wm["empty_flow"] = urwid.Text("")
        wm["empty_box"] = urwid.SolidFill(b" ")
        wm["divider"] = urwid.Divider("─")
        self.canvas = wm.get("canvas", True)

        # Setup menus
        wm.register("menu_file", urwid.Button("File", on_press=self.menu_handler, user_data="menu_file_dropdown"), "menu")
        wm.register("menu_edit", urwid.Button("Edit", on_press=self.menu_handler, user_data="menu_edit_dropdown"), "menu")
        wm.register("menu_view", urwid.Button("View", on_press=self.menu_handler, user_data="menu_view_dropdown"), "menu")
        wm.register("menu_help", urwid.Button("Help", on_press=self.menu_handler, user_data="menu_help_dropdown"), "menu")
        wm.group_add("menu", [(1, wm["empty_flow"]), (8, wm["menu_file"]), (8, wm["menu_edit"]), (8, wm["menu_view"]), (8, wm["menu_help"])], True)
        wm.register("menu", urwid.Columns(wm.group_get("menu"), dividechars=0), "menu")

        # Setup status bar
        wm.register("status", urwid.Text("", align="left"), "status")
        wm.register("time", urwid.Text("", align="right"), "status")
        wm.group_add("footer", [wm["status"], (24, wm["time"])], True)
        wm.register("footer", urwid.Columns(wm.group_get("footer"), dividechars=0), "status")

        # Setup dropdowns
        wm["menu_file_new"] = urwid.Button("New", on_press=self.menu_handler)
        wm["menu_file_open"] = urwid.Button("Open...", on_press=self.menu_handler)
        wm["menu_file_save"] = urwid.Button("Save As...", on_press=self.menu_handler)
        wm["menu_file_imimg"] = urwid.Button("Import Image...", on_press=self.menu_handler)
        wm["menu_file_exbw"] = urwid.Button("Export B&W...", on_press=self.menu_handler)
        wm["menu_file_exclr"] = urwid.Button("Export Color...", on_press=self.menu_handler)
        wm["menu_file_exit"] = urwid.Button("Exit", on_press=self.menu_handler)
        wm.group_add("menu_file", [wm["menu_file_new"], wm["menu_file_open"], wm["menu_file_save"], wm["divider"], wm["menu_file_imimg"], wm["menu_file_exbw"], wm["menu_file_exclr"], wm["divider"], wm["menu_file_exit"]], True)
        wm["menu_file_pile"] = urwid.Pile(wm.group_get("menu_file"))
        wm.register("menu_file_box", urwid.LineBox(wm["menu_file_pile"], ""), "menu")
        wm["menu_file_dropdown"] = urwid.Overlay(wm["menu_file_box"], wm["canvas"], align="left", width=22, valign="top", height="pack", left=1)

        wm["menu_edit_cut"] = urwid.Button("Cut", on_press=self.menu_handler)
        wm["menu_edit_copy"] = urwid.Button("Copy", on_press=self.menu_handler)
        wm["menu_edit_clear"] = urwid.Button("Clear", on_press=self.menu_handler)
        wm["menu_edit_select"] = urwid.Button("Select All", on_press=self.menu_handler)
        wm.group_add("menu_edit", [wm["menu_edit_cut"], wm["menu_edit_copy"], wm["menu_edit_clear"], wm["divider"], wm["menu_edit_select"]], True)
        wm["menu_edit_pile"] = urwid.Pile(wm.group_get("menu_edit"))
        wm.register("menu_edit_box", urwid.LineBox(wm["menu_edit_pile"], ""), "menu")
        wm["menu_edit_dropdown"] = urwid.Overlay(wm["menu_edit_box"], wm["canvas"], align="left", width=16, valign="top", height="pack", left=9)

        wm["menu_view_show"] = urwid.Button("Show toolbox", on_press=self.menu_handler)
        wm["menu_view_hide"] = urwid.Button("Hide toolbox", on_press=self.menu_handler)
        wm.group_add("menu_view", [wm["menu_view_show"], wm["menu_view_hide"]], True)
        wm["menu_view_pile"] = urwid.Pile(wm.group_get("menu_view"))
        wm.register("menu_view_box", urwid.LineBox(wm["menu_view_pile"], ""), "menu")
        wm["menu_view_dropdown"] = urwid.Overlay(wm["menu_view_box"], wm["canvas"], align="left", width=19, valign="top", height="pack", left=15)

        wm["menu_help_about"] = urwid.Button("About...", on_press=self.menu_handler)
        wm["menu_help_help"] = urwid.Button("Instructions...", on_press=self.menu_handler)
        wm.group_add("menu_help", [wm["menu_help_help"], wm["menu_help_about"]], True)
        wm["menu_help_pile"] = urwid.Pile(wm.group_get("menu_help"))
        wm.register("menu_help_box", urwid.LineBox(wm["menu_help_pile"], ""), "menu")
        wm["menu_help_dropdown"] = urwid.Overlay(wm["menu_help_box"], wm["canvas"], align="left", width=22, valign="top", height="pack", left=25)

        # Setup toolbox
        select_group = []
        for group_idx, tool_group in enumerate(DrawCanvas.TOOLS):
            for tool in tool_group:
                wm["tool_{}_button".format(tool)] = urwid.RadioButton(select_group, tool, on_state_change=self.tool_handler)

        wm.group_add("tools", select_group, True)
        wm["tool_grid"] = urwid.GridFlow(wm.group_get("tools"), 10, 1, 0, "center")
        wm["tool_box"] = urwid.LineBox(wm["tool_grid"], "Tools")

        # Setup tool options
        wm.register("options_Paint_1_size", urwid.Edit(edit_text="1"), "option-edit")
        urwid.connect_signal(wm.get("options_Paint_1_size", True), "change", self.option_handler)
        wm["options_Paint_1_box"] = urwid.LineBox(wm["options_Paint_1_size"], "Brush Size")
        wm["options_Paint_1"] = urwid.BoxAdapter(urwid.Filler(wm["options_Paint_1_box"], valign="top"), 4)
        wm.group_add("options_Paint", wm["options_Paint_1"])

        wm.register("options_Erase_1_size", urwid.Edit(edit_text="1"), "option-edit")
        urwid.connect_signal(wm.get("options_Erase_1_size", True), "change", self.option_handler)
        wm["options_Erase_1_box"] = urwid.LineBox(wm["options_Erase_1_size"], "Brush Size")
        wm["options_Erase_1"] = urwid.BoxAdapter(urwid.Filler(wm["options_Erase_1_box"], valign="top"), 4)
        wm.group_add("options_Erase", wm["options_Erase_1"])

        wm["options_Draw_1_symbols"] = urwid.BoxAdapter(SymbolSelect(wm, self.SYMBOLS), 6)
        wm["options_Draw_1_box"] = urwid.LineBox(wm["options_Draw_1_symbols"], "Symbols")
        wm["options_Draw_1"] = urwid.BoxAdapter(urwid.Filler(wm["options_Draw_1_box"], valign="top"), 9)
        wm.group_add("options_Draw", wm["options_Draw_1"])

        wm.register("options_Draw_2_size", urwid.Edit(edit_text="1"), "option-edit")
        urwid.connect_signal(wm.get("options_Draw_2_size", True), "change", self.option_handler)
        wm["options_Draw_2_box"] = urwid.LineBox(wm["options_Draw_2_size"], "Brush Size")
        wm["options_Draw_2"] = urwid.BoxAdapter(urwid.Filler(wm["options_Draw_2_box"], valign="top"), 4)
        wm.group_add("options_Draw", wm["options_Draw_2"])

        stamp_group = []
        wm["options_Stamp_1_original"] = urwid.RadioButton(stamp_group, "Original Colors", on_state_change=self.option_handler)
        wm["options_Stamp_1_current"] = urwid.RadioButton(stamp_group, "Active Colors", on_state_change=self.option_handler)
        wm.group_add("options_Stamp_1_buttons", stamp_group, True)
        wm["options_Stamp_1_grid"] = urwid.GridFlow(wm.group_get("options_Stamp_1_buttons"), 19, 1, 0, "center")
        wm["options_Stamp_1_box"] = urwid.LineBox(wm["options_Stamp_1_grid"], "Stamp Palette")
        wm["options_Stamp_1"] = urwid.BoxAdapter(urwid.Filler(wm["options_Stamp_1_box"], valign="top"), 5)
        wm.group_add("options_Stamp", wm["options_Stamp_1"])

        wm.register("options_Layer_1_current", urwid.Edit(edit_text="0"), "option-edit")
        urwid.connect_signal(wm.get("options_Layer_1_current", True), "change", self.option_handler)
        wm["options_Layer_1_box"] = urwid.LineBox(wm["options_Layer_1_current"], "Current [0-0]")
        wm["options_Layer_1"] = urwid.BoxAdapter(urwid.Filler(wm["options_Layer_1_box"], valign="top"), 4)
        wm.group_add("options_Layer", wm["options_Layer_1"])

        select_group = []
        wm["options_Layer_2_visible"] = urwid.RadioButton(select_group, "Visible", on_state_change=self.option_handler)
        wm["options_Layer_2_hidden"] = urwid.RadioButton(select_group, "Hidden", on_state_change=self.option_handler)
        wm.group_add("options_Layer_2_buttons", select_group, True)
        wm["options_Layer_2_grid"] = urwid.GridFlow(wm.group_get("options_Layer_2_buttons"), 14, 1, 0, "center")
        wm["options_Layer_2_box"] = urwid.LineBox(wm["options_Layer_2_grid"], "Visibility")
        wm["options_Layer_2"] = urwid.BoxAdapter(urwid.Filler(wm["options_Layer_2_box"], valign="top"), 5)
        wm.group_add("options_Layer", wm["options_Layer_2"])

        wm.register("options_Layer_3_add", urwid.Button("  Add  Layer", on_press=self.option_handler), "option-edit")
        wm["options_Layer_3_padding"] = urwid.Padding(wm["options_Layer_3_add"], left=2, right=2)
        wm["options_Layer_3"] = urwid.BoxAdapter(urwid.Filler(wm["options_Layer_3_padding"], valign="top"), 2)
        wm.group_add("options_Layer", wm["options_Layer_3"])

        wm.register("options_Layer_4_del", urwid.Button("  Del  Layer", on_press=self.option_handler), "option-edit")
        wm["options_Layer_4_padding"] = urwid.Padding(wm["options_Layer_4_del"], left=2, right=2)
        wm["options_Layer_4"] = urwid.BoxAdapter(urwid.Filler(wm["options_Layer_4_padding"], valign="top"), 2)
        wm.group_add("options_Layer", wm["options_Layer_4"])

        wm.register("options_Layer_5_flat", urwid.Button(" Flatten Image", on_press=self.option_handler), "option-edit")
        wm["options_Layer_5_padding"] = urwid.Padding(wm["options_Layer_5_flat"], left=2, right=2)
        wm["options_Layer_5"] = urwid.BoxAdapter(urwid.Filler(wm["options_Layer_5_padding"], valign="top"), 2)
        wm.group_add("options_Layer", wm["options_Layer_5"])

        wm.register("options_Image_1_width", urwid.Edit(edit_text=str(wm.get("canvas", True).get_dims()[1])), "option-edit")
        wm["options_Image_1_box"] = urwid.LineBox(wm["options_Image_1_width"], "Width")
        wm["options_Image_1"] = urwid.BoxAdapter(urwid.Filler(wm["options_Image_1_box"], valign="top"), 4)
        wm.group_add("options_Image", wm["options_Image_1"])

        wm.register("options_Image_2_height", urwid.Edit(edit_text=str(wm.get("canvas", True).get_dims()[2])), "option-edit")
        wm["options_Image_2_box"] = urwid.LineBox(wm["options_Image_2_height"], "Height")
        wm["options_Image_2"] = urwid.BoxAdapter(urwid.Filler(wm["options_Image_2_box"], valign="top"), 4)
        wm.group_add("options_Image", wm["options_Image_2"])

        wm.register("options_Image_4_reset", urwid.Button("     Reset", on_press=self.option_handler), "option-edit")
        wm["options_Image_4_padding"] = urwid.Padding(wm["options_Image_4_reset"], left=2, right=2)
        wm["options_Image_4"] = urwid.BoxAdapter(urwid.Filler(wm["options_Image_4_padding"], valign="top"), 2)
        wm.group_add("options_Image", wm["options_Image_4"])

        wm.register("options_Image_5_apply", urwid.Button("     Apply", on_press=self.option_handler), "option-edit")
        wm["options_Image_5_padding"] = urwid.Padding(wm["options_Image_5_apply"], left=2, right=2)
        wm["options_Image_5"] = urwid.BoxAdapter(urwid.Filler(wm["options_Image_5_padding"], valign="top"), 2)
        wm.group_add("options_Image", wm["options_Image_5"])

        # Setup color selection
        wm["fore_button"] = urwid.BoxAdapter(ColorSelect(wm, "canvas", 15, 0), 3)
        wm["back_button"] = urwid.BoxAdapter(ColorSelect(wm, "canvas", 0, 1), 3)
        wm["fore_label"] = urwid.Text("Fore", align="center")
        wm["back_label"] = urwid.Text("Back", align="center")
        wm.group_add("color_grid", [wm["empty_flow"], wm["empty_flow"], wm["fore_button"], wm["back_button"], wm["fore_label"], wm["back_label"]], True)
        wm["color_grid"] = urwid.GridFlow(wm.group_get("color_grid"), 8, 3, 0, "center")
        wm["color_box"] = urwid.LineBox(wm["color_grid"], "Active Colors")

        # Setup dialog boxes
        welcome_config = {"title": "NetPaint {}".format(self.VERSION), "msg": self.WELCOME_MSG, "buttons": ["OK"], "space": 0, "height": 16}
        wm.register("dialog_welcome_overlay", DialogBox("dialog_welcome", wm, "canvas", self.dialog_handler, welcome_config), "dialog")
        open_config = {"title": "Open File", "msg": "Enter filename path", "edit": "", "buttons": ["Open", "Cancel"]}
        wm.register("dialog_open_overlay", DialogBox("dialog_open", wm, "canvas", self.dialog_handler, open_config), "dialog")
        save_config = {"title": "Save File", "msg": "Enter filename path", "edit": "", "buttons": ["Save", "Cancel"]}
        wm.register("dialog_save_overlay", DialogBox("dialog_save", wm, "canvas", self.dialog_handler, save_config), "dialog")
        importimg_config = {"title": "Import Image (Unstable!)", "msg": "Enter filename path", "edit": "", "buttons": ["Open", "Cancel"]}
        wm.register("dialog_importimg_overlay", DialogBox("dialog_importimg", wm, "canvas", self.dialog_handler, importimg_config), "dialog")
        exportbw_config = {"title": "Export File", "msg": "Enter filename path", "edit": "", "buttons": ["Save", "Cancel"]}
        wm.register("dialog_exportbw_overlay", DialogBox("dialog_exportbw", wm, "canvas", self.dialog_handler, exportbw_config), "dialog")
        exportcolor_config = {"title": "Export File", "msg": "Enter filename path", "edit": "", "buttons": ["Save", "Cancel"]}
        wm.register("dialog_exportcolor_overlay", DialogBox("dialog_exportcolor", wm, "canvas", self.dialog_handler, exportcolor_config), "dialog")
        help_config = {"width": 70, "height": 20, "title": "NetPaint Help", "msg": self.HELP_MSG, "scroll": 13, "buttons": ["OK"], "space": 0}
        wm.register("dialog_help_overlay", DialogBox("dialog_help", wm, "canvas", self.dialog_handler, help_config), "dialog")

        # Finalize
        wm.register("toolbox", urwid.Pile([], "toolbox"), "toolbox")
        self.select_tool(DrawCanvas.TOOLS[0][0])
        wm["body"] = urwid.Columns([wm["canvas"], (23, wm["toolbox"])])
        wm["screen"] = urwid.Frame(header=wm["menu"], body=wm["body"], footer=wm["footer"])


def main():
    def parse_args():
        """ Parse command line arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument("image", type=str, nargs="?", help="NetPaint image")
        parser.add_argument("-d", type=int, nargs=2, metavar=("cols", "rows"), default=(120, 40), help="Image dimensions")
        parser.add_argument("--version", "-v", action="store_true", help="Dispaly version")

        return parser.parse_args()

    # Parse arguments
    args = parse_args()

    if args.version:
        print("NetPaint {}".format(NetPaint.VERSION))
        sys.exit(0)

    # Initialize and run netpaint
    netpaint = NetPaint.instance()
    netpaint.setup(args)
    netpaint.start()


if __name__ == "__main__":
    main()
