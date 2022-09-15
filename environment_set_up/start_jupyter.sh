#!/bin/bash
##
# This is a script to help with manual startup of the jupyter server
# if you've configured the HPC end manually (see ManualConfig-HPC.md)
#
# If you used auto-conf.py to set up the jupyter server config, you
# should only need to run 'jupyter lab'.
#
source activate hytest
cd $HOME/HyTEST-Tutorials/
HOST=`hostname`
JPORT=$(shuf -i 8400-9400 -n 1)
echo ""
echo ""
echo "Step 1: Wait until this script says the Jupyter server"
echo "        has started. "
echo ""
echo "Step 2: Copy this ssh command into a terminal on your"
echo "        local computer:"
echo ""
echo "        ssh -N -L 8889:$HOST:$JPORT  $USER@$SLURM_CLUSTER_NAME.cr.usgs.gov"
echo ""
echo "Step 3: Browse to http://localhost:8889 on your local computer"
echo ""
echo ""
jupyter lab --no-browser --ip=$HOST --port=$JPORT
