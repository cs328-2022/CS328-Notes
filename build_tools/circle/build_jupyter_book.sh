#!/usr/bin/env bash
set -xe

jupyter_book_dir=CS328-Notes
jupyter_book_build_dir="$jupyter_book_dir/_build/html"

function show_error_logs {
    echo "Some notebooks failed, see logs below:"
    for f in $jupyter_book_build_dir/reports/*.log; do
        echo "================================================================================"
        echo $f
        echo "================================================================================"
        cat $f
    done
    # You need to exit with non-zero here to cause the build to fail
    exit 1
}

apt-get install make

source /opt/conda/etc/profile.d/conda.sh
conda update --yes conda
conda create -n cs328 --yes -c conda-forge python=3.9
conda activate cs328
pip install -r requirements.txt

jupyter-book build CS328-Notes
