#!/usr/bin/env bash

# save current working directory where the test related information is stored
cwd=$(pwd)


display_help() {
    echo "Usage: $0 [-r]" >&2
    echo
    echo "   -r, --report   create line reports"
    echo
    exit 1
}


if [ "$1" == "-h" ] ; then
    echo "Usage: `basename $0` [-h] [-r]"
    exit 0
fi


if [ "$1" == "-h" ] ; then
    echo "Usage: `basename $0` [-h] [-r]"
    exit 0
fi


# run mypy from base directory since otherwise the html report will not be done
cd ..


# check command line parameters
while :
do
    case "$1" in
		-h | --help)
			display_help  # Call your function
            exit 0
            ;;
		-r | --report)
			python3 -m mypy \
			    --no-incremental \
			    --config-file ${cwd}/mypy.ini \
			    --linecount-report ${cwd}/mypy-report \
			    --html-report ${cwd}/mypy-report \
			    --package pde
			exit 0
			;;
		--) # End of all options
			shift
			break
			;;
      -*)
			echo "Error: Unknown option: $1" >&2
          	display_help
          	exit 1 
          	;;
		*)  # No more options
			python3 -m mypy \
			    --config-file ${cwd}/mypy.ini \
			    --pretty \
			    --package pde
			exit 0         	;;			
	esac
done



    
    