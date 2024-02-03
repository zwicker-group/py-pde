echo "Test codestyle"
echo "--------------"
./tests_codestyle.sh

echo ""
echo "Test types"
echo "----------"
./tests_types.sh

echo ""
echo "Run serial unittests"
echo "-------------"
./tests_parallel.sh

echo ""
echo "Run parallel unittests"
echo "-------------"
./tests_mpi.sh