#!/bin/bash

# source ../_private/env.sh
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")
sudo docker-compose -f $scriptDir/infrastructure.yaml up