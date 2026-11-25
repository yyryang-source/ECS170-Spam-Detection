"""Run instructor-style Citeseer GCN."""
import sys
from script_gcn import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--dataset", "citeseer"]
    main()
