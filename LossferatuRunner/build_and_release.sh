#!/bin/bash

swift build -c release
cd .build/release
sudo cp -f LossferatuRunner /usr/local/bin/LossferatuRunner
