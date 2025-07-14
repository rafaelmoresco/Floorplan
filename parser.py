from lefdef import C_LefReader, C_DefReader
from dataStructures import Circuit, Macro, PinLEF, Port, Rect, Obs, Component, PinDEF, Net, Row, Track, GCellGrid, DieArea


def fill_circuit_from_lefdef(circuit: Circuit, lef_path: str, def_path: str):
    lef_reader = C_LefReader()
    lef = lef_reader.read(lef_path)

    # Fill LEF Macros
    for i in range(lef.c_num_macros):
        macro = lef.c_macros[i]
        # Pins
        pins = []
        for j in range(macro.c_num_pins):
            pin = macro.c_pins[j]
            # Ports
            ports = []
            for k in range(pin.c_num_ports):
                port = pin.c_ports[k]
                # Rects
                rects = []
                for l in range(port.c_num_rects):
                    rect = port.c_rects[l]
                    rects.append(Rect(rect.c_layer, rect.c_xl, rect.c_yl, rect.c_xh, rect.c_yh))
                ports.append(Port(rects))
            pins.append(PinLEF(pin.c_name, pin.c_direction, pin.c_use, pin.c_shape, ports))
        # Obs
        obs_rects = []
        for j in range(macro.c_obs.c_num_rects):
            rect = macro.c_obs.rects[j]
            obs_rects.append(Rect(rect.c_layer, rect.c_xl, rect.c_yl, rect.c_xh, rect.c_yh))
        obs = Obs(obs_rects)
        circuit.macros.append(Macro(macro.c_class, macro.c_source, macro.c_site_name, macro.c_origin_x, macro.c_origin_y, macro.c_foreign_name, macro.c_foreign_x, macro.c_foreign_y, macro.c_foreign_orient, pins, obs))

    # Fill DEF
    def_reader = C_DefReader()
    _def = def_reader.read(def_path)

    # Die area
    circuit.die_area = DieArea(_def.c_die_area_width, _def.c_die_area_height)

    # GCell Grids X
    for i in range(_def.c_num_g_cell_grid_x):
        g = _def.c_g_cell_grid_x[i]
        circuit.gcell_grids_x.append(GCellGrid(g.c_offset, g.c_num, g.c_step))
    # GCell Grids Y
    for i in range(_def.c_num_g_cell_grid_y):
        g = _def.c_g_cell_grid_y[i]
        circuit.gcell_grids_y.append(GCellGrid(g.c_offset, g.c_num, g.c_step))

    # Components
    for i in range(_def.c_num_components):
        c = _def.c_components[i]
        circuit.components.append(Component(c.c_id, c.c_name, c.c_status, c.c_source, c.c_orient, c.c_x, c.c_y))

    # Pins
    for i in range(_def.c_num_pins):
        p = _def.c_pins[i]
        # Rects
        rects = []
        for j in range(p.c_num_rects):
            r = p.c_rects[j]
            rects.append(Rect(r.c_layer, r.c_xl, r.c_yl, r.c_xh, r.c_yh))
        # Ports
        ports = []
        for j in range(p.c_num_ports):
            port = p.c_ports[j]
            port_rects = []
            for k in range(port.c_num_rects):
                r = port.c_rects[k]
                port_rects.append(Rect(r.c_layer, r.c_xl, r.c_yl, r.c_xh, r.c_yh))
            ports.append(Port(port_rects))
        circuit.pins_def.append(PinDEF(p.c_name, p.c_net, p.c_use, p.c_status, p.c_direction, p.c_orient, p.c_x, p.c_y, rects, ports))

    # Nets
    for i in range(_def.c_num_nets):
        n = _def.c_nets[i]
        circuit.nets.append(Net(n.c_name, n.c_instances, n.c_pins))

    # Rows
    for i in range(_def.c_num_rows):
        r = _def.c_rows[i]
        circuit.rows.append(Row(r.c_name, r.c_macro, r.c_x, r.c_y, r.c_num_x, r.c_num_y, r.c_step_x, r.c_step_y))

    # Tracks X
    for i in range(_def.c_num_tracks_x):
        t = _def.c_tracks_x[i]
        circuit.tracks_x.append(Track(t.c_layer, t.c_offset, t.c_num, t.c_step))
    # Tracks Y
    for i in range(_def.c_num_tracks_y):
        t = _def.c_tracks_y[i]
        circuit.tracks_y.append(Track(t.c_layer, t.c_offset, t.c_num, t.c_step))
