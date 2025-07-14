from typing import Generic, TypeVar, Optional, List, Tuple
import random

T = TypeVar('T')

class BStarTreeNode(Generic[T]):
    def __init__(self, data: T, left: Optional['BStarTreeNode[T]'] = None, right: Optional['BStarTreeNode[T]'] = None):
        self.data = data
        self.left = left
        self.right = right

class BStarTree(Generic[T]):
    def __init__(self):
        self.root: Optional[BStarTreeNode[T]] = None

    def insert(self, data: T, heuristic: str = "leftmost"):
        """
        Insert a new node into the B* tree using the specified heuristic.
        Heuristics: "leftmost", "rightmost", "random", "balanced"
        """
        if self.root is None:
            self.root = BStarTreeNode(data)
        else:
            if heuristic == "leftmost":
                self._insert_leftmost(self.root, data)
            elif heuristic == "rightmost":
                self._insert_rightmost(self.root, data)
            elif heuristic == "random":
                self._insert_random(self.root, data)
            elif heuristic == "balanced":
                self._insert_balanced(self.root, data)
            else:
                raise ValueError(f"Unknown heuristic: {heuristic}")

    def _insert_leftmost(self, node: BStarTreeNode[T], data: T):
        if node.left is None:
            node.left = BStarTreeNode(data)
        else:
            self._insert_leftmost(node.left, data)

    def _insert_rightmost(self, node: BStarTreeNode[T], data: T):
        if node.right is None:
            node.right = BStarTreeNode(data)
        else:
            self._insert_rightmost(node.right, data)

    def _insert_random(self, node: BStarTreeNode[T], data: T):
        if random.choice([True, False]):  # 50% chance for left or right
            if node.left is None:
                node.left = BStarTreeNode(data)
            else:
                self._insert_random(node.left, data)
        else:
            if node.right is None:
                node.right = BStarTreeNode(data)
            else:
                self._insert_random(node.right, data)

    def _insert_balanced(self, node: BStarTreeNode[T], data: T):
        # Try to maintain balance by choosing the subtree with fewer nodes
        left_count = self._count_nodes(node.left)
        right_count = self._count_nodes(node.right)
        
        if left_count <= right_count:
            if node.left is None:
                node.left = BStarTreeNode(data)
            else:
                self._insert_balanced(node.left, data)
        else:
            if node.right is None:
                node.right = BStarTreeNode(data)
            else:
                self._insert_balanced(node.right, data)

    def _count_nodes(self, node: Optional[BStarTreeNode[T]]) -> int:
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def inorder_traversal(self) -> List[T]:
        result: List[T] = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: Optional[BStarTreeNode[T]], result: List[T]):
        if node is not None:
            self._inorder(node.left, result)
            result.append(node.data)
            self._inorder(node.right, result)

    def build(self):
        pass

    def swap(self, data1: T, data2: T) -> bool:
        """Swap the data of two nodes containing data1 and data2. Returns True if successful, False if either not found."""
        node1 = self._find_node(self.root, data1)
        node2 = self._find_node(self.root, data2)
        if node1 is not None and node2 is not None:
            node1.data, node2.data = node2.data, node1.data
            return True
        return False

    def _find_node(self, node: Optional[BStarTreeNode[T]], data: T) -> Optional[BStarTreeNode[T]]:
        if node is None:
            return None
        if node.data == data:
            return node
        left_result = self._find_node(node.left, data)
        if left_result is not None:
            return left_result
        return self._find_node(node.right, data)

    def swap_nodes(self, data1: T, data2: T) -> bool:
        """Swap the actual nodes (not just data) containing data1 and data2. Returns True if successful, False if either not found."""
        if data1 == data2:
            return True  # Nothing to do
        parent1, node1, is_left1 = self._find_with_parent(self.root, None, data1)
        parent2, node2, is_left2 = self._find_with_parent(self.root, None, data2)
        if node1 is None or node2 is None or node1 is node2:
            return False
        # If one node is parent of the other, special handling is needed
        if node1 is parent2:  # node1 is parent of node2
            self._swap_parent_child(parent1, node1, is_left1, node2, is_left2)
        elif node2 is parent1:  # node2 is parent of node1
            self._swap_parent_child(parent2, node2, is_left2, node1, is_left1)
        else:
            # Swap parent pointers
            if parent1 is None:
                self.root = node2
            else:
                if is_left1:
                    parent1.left = node2
                else:
                    parent1.right = node2
            if parent2 is None:
                self.root = node1
            else:
                if is_left2:
                    parent2.left = node1
                else:
                    parent2.right = node1
            # Swap children
            node1.left, node2.left = node2.left, node1.left
            node1.right, node2.right = node2.right, node1.right
        return True

    def _find_with_parent(self, node: Optional[BStarTreeNode[T]], parent: Optional[BStarTreeNode[T]], data: T):
        if node is None:
            return None, None, False
        if node.data == data:
            is_left = parent.left == node if parent else False
            return parent, node, is_left
        left_result = self._find_with_parent(node.left, node, data)
        if left_result[1] is not None:
            return left_result
        return self._find_with_parent(node.right, node, data)

    def _swap_parent_child(self, parent, parent_node, is_left_parent, child_node, is_left_child):
        # parent_node is parent of child_node
        # Update parent's pointer
        if parent is None:
            self.root = child_node
        else:
            if is_left_parent:
                parent.left = child_node
            else:
                parent.right = child_node
        # Swap children
        if is_left_child:
            child_node.left, parent_node.left = parent_node, child_node.left
            child_node.right, parent_node.right = parent_node.right, child_node.right
        else:
            child_node.right, parent_node.right = parent_node, child_node.right
            child_node.left, parent_node.left = parent_node.left, child_node.left

    def rotate_macro(self, macro: T):
        """Rotate the macro and toggle its 'rotated' attribute."""
        node = self._find_node(self.root, macro)
        if node is not None and hasattr(node.data, 'rotated'):
            node.data.rotated = not node.data.rotated
            return True
        return False

    def move_macro_to_random(self, macro: T):
        """Remove the macro from the tree and reinsert it at a random position."""
        # Remove macro from tree
        self._remove_macro(macro)
        # Reinsert at random position
        self.insert(macro, heuristic="random")

    def _remove_macro(self, macro: T):
        """Remove the macro from the tree (by data)."""
        self.root = self._remove_node(self.root, macro)

    def _remove_node(self, node: Optional[BStarTreeNode[T]], macro: T) -> Optional[BStarTreeNode[T]]:
        if node is None:
            return None
        if node.data == macro:
            # Remove this node, reattach children
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            # Node with two children: replace with leftmost of right subtree
            min_larger_node = node.right
            while min_larger_node.left:
                min_larger_node = min_larger_node.left
            node.data = min_larger_node.data
            node.right = self._remove_node(node.right, min_larger_node.data)
            return node
        node.left = self._remove_node(node.left, macro)
        node.right = self._remove_node(node.right, macro)
        return node

    def copy(self) -> 'BStarTree[T]':
        """Create a deep copy of this B* tree."""
        new_tree = BStarTree()
        if self.root:
            new_tree.root = self._copy_node(self.root)
        return new_tree

    def _copy_node(self, node: Optional[BStarTreeNode[T]]):
        """Recursively copy a B* tree node."""
        if node is None:
            return None
        new_node = BStarTreeNode(node.data)
        new_node.left = self._copy_node(node.left)
        new_node.right = self._copy_node(node.right)
        return new_node

    def collect_nodes(self) -> list:
        """Collect all nodes in the tree into a list."""
        nodes_list = []
        def _collect(node):
            if node is None:
                return
            nodes_list.append(node)
            _collect(node.left)
            _collect(node.right)
        _collect(self.root)
        return nodes_list

    def swap_random_nodes(self):
        """Swap two random nodes in the B* tree."""
        if not self.root:
            return
        nodes = self.collect_nodes()
        if len(nodes) < 2:
            return
        node1, node2 = random.sample(nodes, 2)
        node1.data, node2.data = node2.data, node1.data

class Circuit():
    def __init__(self):
        # LEF objects
        self.macros: List[Macro] = []
        self.pins_lef: List[PinLEF] = []
        self.ports: List[Port] = []
        self.rects: List[Rect] = []
        self.obs: List[Obs] = []
        # DEF objects
        self.components: List[Component] = []
        self.pins_def: List[PinDEF] = []
        self.nets: List[Net] = []
        self.rows: List[Row] = []
        self.tracks_x: List[Track] = []
        self.tracks_y: List[Track] = []
        self.gcell_grids_x: List[GCellGrid] = []
        self.gcell_grids_y: List[GCellGrid] = []
        self.die_area: Optional[DieArea] = None
        # B* Tree
        self.bstar_tree: BStarTree[Macro] = BStarTree()
        # Contours
        self.contour: List[Tuple[int, int]] = [(0, 0)]
        self.vcontour: List[Tuple[int, int]] = [(0, 0)]

    def find_max_y(self,xl, xh):
        max_y = 0
        for i in range(len(self.contour) - 1):
            seg_xl, seg_y = self.contour[i]
            seg_xh, _ = self.contour[i + 1]
            if seg_xh <= xl or seg_xl >= xh:
                continue
            max_y = max(max_y, seg_y)
        return max_y

    def update_contour(self, xl, xh, top_y):
        new_contour = []
        i = 0
        while i < len(self.contour) and self.contour[i][0] < xl:
            new_contour.append(self.contour[i])
            i += 1
        if not new_contour or new_contour[-1][0] < xl:
            new_contour.append((xl, self.find_max_y(xl, xl)))
        new_contour.append((xl, top_y))
        while i < len(self.contour) and self.contour[i][0] <= xh:
            i += 1
        prev_y = self.find_max_y(xh, xh)
        new_contour.append((xh, prev_y))
        new_contour.extend(self.contour[i:])
        deduped = [new_contour[0]]
        for pt in new_contour[1:]:
            if pt[1] != deduped[-1][1]:
                deduped.append(pt)
        return deduped

    def find_max_x(self, yl, yh):
        max_x = 0
        for i in range(len(self.vcontour) - 1):
            seg_yl, seg_x = self.vcontour[i]
            seg_yh, _ = self.vcontour[i + 1]
            if seg_yh <= yl or seg_yl >= yh:
                continue
            max_x = max(max_x, seg_x)
        return max_x

    def update_vcontour(self, yl, yh, right_x):
        new_vcontour = []
        i = 0
        while i < len(self.vcontour) and self.vcontour[i][0] < yl:
            new_vcontour.append(self.vcontour[i])
            i += 1
        if not new_vcontour or new_vcontour[-1][0] < yl:
            new_vcontour.append((yl, self.find_max_x(yl, yl)))
        new_vcontour.append((yl, right_x))
        while i < len(self.vcontour) and self.vcontour[i][0] <= yh:
            i += 1
        prev_x = self.find_max_x(yh, yh)
        new_vcontour.append((yh, prev_x))
        new_vcontour.extend(self.vcontour[i:])
        deduped = [new_vcontour[0]]
        for pt in new_vcontour[1:]:
            if pt[1] != deduped[-1][1]:
                deduped.append(pt)
        return deduped
    
    def bstar_to_circuit(self, debug=False):
        """
        Place macros using B*-tree contour packing with both horizontal and vertical contours.
        """
        if not self.bstar_tree.root:
            return
        self.contour = [(0, 0)]
        self.vcontour = [(0, 0)]

        def place(node, x, position):
            if node is None:
                return
            macro = node.data
            width = macro.get_macro_width()
            height = macro.get_macro_height()
            xl = x
            xh = x + width
            # Default: place at lowest y using horizontal contour
            y = self.find_max_y(xl, xh)
            if position == "Above":
                # For right child, use vertical contour to find rightmost x
                xl = self.find_max_x(y, y + height)
                xh = xl + width
            macro.origin_x = xl
            macro.origin_y = y
            self.contour = self.update_contour(xl, xh, y + height)
            self.vcontour = self.update_vcontour(y, y + height, xl + width)
            if debug:
                name = getattr(macro, 'site_name', None) or getattr(macro, 'class_', None) or str(macro)
                print(f"Placing macro: {name} at ({xl},{y}) size=({width},{height}) {position}")
                #print(f"Contour: {self.contour}")
                #print(f"VContour: {self.vcontour}")
            # Place left child to the right
            if node.left:
                place(node.left, xh, "Right")
            # Place right child above parent (at same x)
            if node.right:
                place(node.right, xl, "Above")

        place(self.bstar_tree.root, 0, "Start")

        self.check_overlaps(debug=debug)

    def check_overlaps(self, debug=False):
        """Check for overlaps between macros and print warnings if found."""
        macros = self.macros
        for i in range(len(macros)):
            m1 = macros[i]
            x1, y1 = m1.origin_x, m1.origin_y
            w1, h1 = m1.get_macro_width(), m1.get_macro_height()
            for j in range(i+1, len(macros)):
                m2 = macros[j]
                x2, y2 = m2.origin_x, m2.origin_y
                w2, h2 = m2.get_macro_width(), m2.get_macro_height()
                if (x1 < x2 + w2 and x1 + w1 > x2 and
                    y1 < y2 + h2 and y1 + h1 > y2):
                    if debug:
                        n1 = getattr(m1, 'site_name', None) or getattr(m1, 'class_', None) or str(m1)
                        n2 = getattr(m2, 'site_name', None) or getattr(m2, 'class_', None) or str(m2)
                        print(f"WARNING: Overlap detected between {n1} and {n2}")
                    return True
        return False

    def circuit_to_bstar_random(self, seed: int = 42):
        """
        Randomly fill a B* tree from the available macros in the circuit.
        Uses the provided seed for reproducibility.
        """
        random.seed(seed)
        self.bstar_tree = BStarTree()
        if not self.macros:
            return
        # Randomly shuffle the macros
        shuffled_macros = self.macros.copy()
        random.shuffle(shuffled_macros)
        # Insert each macro into the B* tree
        for macro in shuffled_macros:
            self.bstar_tree.insert(macro, heuristic="balanced")

    def get_floorplan_area(self):
        """
        Returns (width, height, area) of the floorplan, based on macro positions and sizes.
        """
        if not self.macros:
            return 0, 0, 0
        max_x = 0
        max_y = 0
        for macro in self.macros:
            # Use the same width/height logic as in bstar_to_circuit
            x = macro.origin_x + macro.get_macro_width()
            y = macro.origin_y + macro.get_macro_height()
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        area = (max_x * max_y) / 1000000 # Convert to mm^2
        return max_x / 1000, max_y / 1000, area

    def plot_floorplan(self, figsize=(10, 10), show_names=True, save_path=None):
        """
        Plot the positioned macros of the circuit using matplotlib.
        Each macro is shown as a rectangle with its name.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=figsize)
        for macro in self.macros:
            # Get macro bounding box
            if macro.obs and macro.obs.rects:
                rect = macro.obs.rects[0]
                width = macro.get_macro_width()
                height = macro.get_macro_height()
            elif macro.pins and macro.pins[0].ports and macro.pins[0].ports[0].rects:
                all_x = [r.xl for p in macro.pins for port in p.ports for r in port.rects] + [r.xh for p in macro.pins for port in p.ports for r in port.rects]
                all_y = [r.yl for p in macro.pins for port in p.ports for r in port.rects] + [r.yh for p in macro.pins for port in p.ports for r in port.rects]
                width = max(all_x) - min(all_x)
                height = max(all_y) - min(all_y)
            else:
                width = height = 0
            x = macro.origin_x
            y = macro.origin_y
            # Draw rectangle
            rect_patch = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect_patch)
            # Draw macro name
            if show_names:
                name = getattr(macro, 'site_name', None) or getattr(macro, 'class_', None) or ''
                ax.text(x + width/2, y + height/2, name, ha='center', va='center', fontsize=8, color='red')
        # Set axis limits
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_xlabel('X (microns)')
        ax.set_ylabel('Y (microns)')
        ax.set_title('Circuit Floorplan')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def print_bstar_tree(self):
        """Print the current B* tree structure for debugging."""
        def get_macro_name(macro):
            return getattr(macro, 'site_name', None) or getattr(macro, 'class_', None) or str(macro)
        def print_node(node, prefix="", is_left=None):
            if node is None:
                return
            macro = node.data
            name = get_macro_name(macro)
            connector = ""
            if is_left is True:
                connector = "├─L─ "
            elif is_left is False:
                connector = "└─R─ "
            print(f"{prefix}{connector}{name}")
            if node.left or node.right:
                if node.left:
                    print_node(node.left, prefix + ("│   " if node.right else "    "), True)
                if node.right:
                    print_node(node.right, prefix + "    ", False)
        print("B* Tree Structure:")
        print_node(self.bstar_tree.root)

# LEF Classes
class Rect:
    def __init__(self, c_layer: str, c_xl: int, c_yl: int, c_xh: int, c_yh: int):
        self.layer = c_layer
        self.xl = c_xl
        self.yl = c_yl
        self.xh = c_xh
        self.yh = c_yh

class Port:
    def __init__(self, c_rects: list):
        self.rects = c_rects  # List[Rect]

class PinLEF:
    def __init__(self, c_name: str, c_direction: str, c_use: str, c_shape: str, c_ports: list):
        self.name = c_name
        self.direction = c_direction
        self.use = c_use
        self.shape = c_shape
        self.ports = c_ports  # List[Port]

class Obs:
    def __init__(self, c_rects: list):
        self.rects = c_rects  # List[Rect]

class Macro:
    def __init__(self, c_class: str, c_source: str, c_site_name: str, c_origin_x: int, c_origin_y: int,
                 c_foreign_name: str, c_foreign_x: int, c_foreign_y: int, c_foreign_orient: str,
                 c_pins: list, c_obs: Obs):
        self.class_ = c_class
        self.source = c_source
        self.site_name = c_site_name
        self.origin_x = c_origin_x
        self.origin_y = c_origin_y
        self.foreign_name = c_foreign_name
        self.foreign_x = c_foreign_x
        self.foreign_y = c_foreign_y
        self.foreign_orient = c_foreign_orient
        self.pins = c_pins  # List[PinLEF]
        self.obs = c_obs
        self.rotated = False  # Add rotated attribute

    def get_macro_width(self):
        if self.rotated:
            return self._get_macro_height_raw()
        return self._get_macro_width_raw()

    def get_macro_height(self):
        if self.rotated:
            return self._get_macro_width_raw()
        return self._get_macro_height_raw()

    def _get_macro_width_raw(self):
        if self.obs and self.obs.rects:
            return max(r.xh for r in self.obs.rects) - min(r.xl for r in self.obs.rects)
        elif self.pins and self.pins[0].ports and self.pins[0].ports[0].rects:
            return max(r.xh for p in self.pins for port in p.ports for r in port.rects) - min(r.xl for p in self.pins for port in p.ports for r in port.rects)
        else:
            return 0

    def _get_macro_height_raw(self):
        if self.obs and self.obs.rects:
            return max(r.yh for r in self.obs.rects) - min(r.yl for r in self.obs.rects)
        elif self.pins and self.pins[0].ports and self.pins[0].ports[0].rects:
            return max(r.yh for p in self.pins for port in p.ports for r in port.rects) - min(r.yl for p in self.pins for port in p.ports for r in port.rects)
        else:
            return 0
# DEF Classes
class Component:
    def __init__(self, c_id: str, c_name: str, c_status: str, c_source: str, c_orient: str, c_x: int, c_y: int):
        self.id = c_id
        self.name = c_name
        self.status = c_status
        self.source = c_source
        self.orient = c_orient
        self.x = c_x
        self.y = c_y

class PinDEF:
    def __init__(self, c_name: str, c_net: str, c_use: str, c_status: str, c_direction: str, c_orient: str, c_x: int, c_y: int, c_rects: list, c_ports: list):
        self.name = c_name
        self.net = c_net
        self.use = c_use
        self.status = c_status
        self.direction = c_direction
        self.orient = c_orient
        self.x = c_x
        self.y = c_y
        self.rects = c_rects  # List[Rect]
        self.ports = c_ports  # List[Port]

class Net:
    def __init__(self, c_name: str, c_instances: list, c_pins: list):
        self.name = c_name
        self.instances = c_instances  # List[str]
        self.pins = c_pins  # List[str]

class Row:
    def __init__(self, c_name: str, c_macro: str, c_x: int, c_y: int, c_num_x: int, c_num_y: int, c_step_x: int, c_step_y: int):
        self.name = c_name
        self.macro = c_macro
        self.x = c_x
        self.y = c_y
        self.num_x = c_num_x
        self.num_y = c_num_y
        self.step_x = c_step_x
        self.step_y = c_step_y

class Track:
    def __init__(self, c_layer: str, c_offset: int, c_num: int, c_step: int):
        self.layer = c_layer
        self.offset = c_offset
        self.num = c_num
        self.step = c_step

class GCellGrid:
    def __init__(self, c_offset: int, c_num: int, c_step: int):
        self.offset = c_offset
        self.num = c_num
        self.step = c_step

class DieArea:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height