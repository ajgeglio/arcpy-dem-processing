#!/bin/bash

# Define paths to your files relative to this script
WARRANTY_FILE="./warranty"
LICENSE_FILE="./license"

case "$1" in
  show-w)
    if [[ -f "$WARRANTY_FILE" ]]; then
      cat "$WARRANTY_FILE"
    else
      echo "Warranty file not found."
    fi
    ;;
  show-c)
    if [[ -f "$LICENSE_FILE" ]]; then
      cat "$LICENSE_FILE"
    else
      echo "License file not found."
    fi
    ;;
  *)
    echo "Usage: $0 {show-w|show-c}"
    ;;
esac