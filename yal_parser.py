import re
from dataStructures import Circuit, Macro, PinLEF, Port, Rect, Obs, Component, PinDEF, Net

def parse_yal_file(yal_path: str, circuit: Circuit):
    """
    Parse a YAL file and fill the given Circuit object.
    """
    with open(yal_path, 'r') as f:
        yal_text = f.read()

    # Remove comments (/* ... */)
    yal_text = re.sub(r'/\*.*?\*/', '', yal_text, flags=re.DOTALL)
    # Replace semicolons with newlines for easier splitting
    yal_text = yal_text.replace(';', '\n')
    # Remove extra whitespace
    yal_text = re.sub(r'\s+', ' ', yal_text)

    # Tokenize by MODULE
    modules = re.split(r'\bMODULE\b', yal_text)
    for module_block in modules[1:]:  # skip the first split (before first MODULE)
        module_block = module_block.strip()
        if not module_block:
            continue
        # Get module name
        m = re.match(r'([\w\d_\-\.]+)', module_block)
        if not m:
            continue
        module_name = m.group(1)
        block = module_block[m.end():].strip()

        # Extract TYPE
        type_match = re.search(r'TYPE ([A-Z]+)', block)
        module_type = type_match.group(1) if type_match else None

        # Extract DIMENSIONS (if present)
        dim_match = re.search(r'DIMENSIONS ([\d\.\s]+)', block)
        dims = []
        if dim_match:
            dims = [float(x) for x in dim_match.group(1).split()]
            # dims: [x1, y1, x2, y2, ...]

        # Extract IOLIST
        iolist_match = re.search(r'IOLIST (.*?) ENDIOLIST', block)
        iolist = []
        if iolist_match:
            iolist_lines = iolist_match.group(1).strip().split('  ')
            for line in iolist_lines:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                # Try to parse as much as possible
                signal = tokens[0]
                term_type = tokens[1]
                # Try to get x, y, width, layer if present
                x = y = width = layer = None
                for t in tokens[2:]:
                    try:
                        val = float(t)
                        if x is None:
                            x = val
                        elif y is None:
                            y = val
                        elif width is None:
                            width = val
                    except ValueError:
                        if layer is None and t.isalpha():
                            layer = t
                iolist.append({
                    'signal': signal,
                    'type': term_type,
                    'x': x, 'y': y, 'width': width, 'layer': layer
                })

        # Extract NETWORK (if present)
        network_match = re.search(r'NETWORK (.*?) ENDNETWORK', block)
        network = []
        if network_match:
            net_lines = network_match.group(1).strip().split('  ')
            for line in net_lines:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                inst_name = tokens[0]
                mod_name = tokens[1]
                signals = tokens[2:]
                network.append({'inst': inst_name, 'mod': mod_name, 'signals': signals})

        # Extract PLACEMENT (if present)
        placement_match = re.search(r'PLACEMENT (.*?) ENDPLACEMENT', block)
        placement = []
        if placement_match:
            place_lines = placement_match.group(1).strip().split('  ')
            for line in place_lines:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                inst_name = tokens[0]
                xloc = float(tokens[1])
                yloc = float(tokens[2])
                placement.append({'inst': inst_name, 'x': xloc, 'y': yloc})

        # Now, map to Circuit data structures
        # For primitive cells (STANDARD, PAD, GENERAL, FEEDTHROUGH):
        if module_type in ('STANDARD', 'PAD', 'GENERAL', 'FEEDTHROUGH'):
            # Use DIMENSIONS to get bounding box
            if dims and len(dims) >= 4:
                min_x = min(dims[::2])
                max_x = max(dims[::2])
                min_y = min(dims[1::2])
                max_y = max(dims[1::2])
            else:
                min_x = min_y = max_x = max_y = 0
            obs_rect = Rect('YAL', min_x, min_y, max_x, max_y)
            obs = Obs([obs_rect])
            # Pins
            pins = []
            for io in iolist:
                # Only add if x/y/width/layer are present
                if io['x'] is not None and io['y'] is not None and io['width'] is not None and io['layer'] is not None:
                    rect = Rect(io['layer'], io['x']-io['width']/2, io['y']-io['width']/2, io['x']+io['width']/2, io['y']+io['width']/2)
                    port = Port([rect])
                    pins.append(PinLEF(io['signal'], io['type'], '', '', [port]))
            macro = Macro(module_type, '', module_name, 0, 0, '', 0, 0, '', pins, obs)
            circuit.macros.append(macro)
        # For PARENT modules: create components, pins, nets
        elif module_type == 'PARENT':
            # Components from NETWORK
            for net in network:
                comp = Component(net['inst'], net['mod'], '', '', '', 0, 0)
                circuit.components.append(comp)
            # Pins from IOLIST
            for io in iolist:
                # Only add if side/position or x/y present
                name = io['signal']
                net = ''
                use = io['type']
                status = ''
                direction = io['type']
                orient = ''
                x = io['x'] if io['x'] is not None else 0
                y = io['y'] if io['y'] is not None else 0
                rects = []
                ports = []
                circuit.pins_def.append(PinDEF(name, net, use, status, direction, orient, x, y, rects, ports))
            # Nets from NETWORK
            for net in network:
                circuit.nets.append(Net(net['inst'], [net['mod']], net['signals']))
