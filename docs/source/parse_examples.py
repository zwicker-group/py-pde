#!/usr/bin/env python3

import pathlib

# Root direcotry of the package
ROOT = pathlib.Path(__file__).absolute().parents[2]
# directory where all the examples reside
INPUT = ROOT / 'examples'
# directory to which the documents are writen
OUTPUT = ROOT / 'docs' / 'source' / 'examples'



def main():
    """ parse all examples and write them in a special example module """
    # create the output directory
    OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # iterate over all examples
    for path_in in INPUT.glob("*.py"):
        path_out = OUTPUT / (path_in.stem + ".rst")
        print(f'Found example {path_in}')
        with path_in.open("r") as file_in, path_out.open("w") as file_out:
            # write the header for the rst file
            file_out.write(".. code-block:: python\n\n")
            
            # add the actual code lines 
            header = True
            for line in file_in:
                # skip the shebang, comments and empty lines in the beginning
                if header and (line.startswith("#") or len(line.strip()) == 0):
                    continue
                header = False  # first real line was reached
                file_out.write('    ' + line)
    


if __name__ == '__main__':
    main()