#!/usr/bin/bash

run_main() {
    # Ensure at least a filename is provided
    if [ -z "$1" ]; then
        echo "Error: No Python script specified."
        return 1
    fi

    # Run Python with the first argument (script name)
    # and pass all subsequent arguments ("${@:2}")
    python "$1" "${@:2}"

    ret=$?

    if [ $ret -ne 0 ]; then
        echo "Python script '$1' exited with an error code: $ret"
        # Use return instead of exit inside functions so you don't close
        # the entire terminal or parent shell session unexpectedly.
        return $ret
    fi
}

### We no longer care about torch or torchao
# run_main main.py pytorch pytorch
# sleep 60
# run_main main.py pytorch ao
# sleep 60
run_main main.py sgl-project sglang || exit 1
sleep 60
run_main main.py vllm-project vllm || exit 1
