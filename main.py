from parser import fill_circuit_from_lefdef
from yal_parser import parse_yal_file
from annealing import *
from dataStructures import *
import datetime

def main():
    circuit = Circuit()
    parse_yal_file("data/ami49.yal", circuit)
    annealer = Annealer(circuit)
    synttime = []
    area = []
    for i in range(5):
        start_time = datetime.datetime.now()
        annealer.anneal()
        end_time = datetime.datetime.now()
        time_taken = end_time - start_time
        synttime.append(time_taken)
        area.append(circuit.get_floorplan_area()[2])
        print(f"Annealing completed in: {time_taken}")
        print(area[i])
    #print(circuit.bstar_tree.inorder_traversal())
    #circuit.bstar_to_circuit()
    #circuit.print_bstar_tree()
    #circuit.plot_floorplan(save_path="data/ami49_floorplan.png")
    print(synttime)
    print(area)

if __name__ == "__main__":
    main()