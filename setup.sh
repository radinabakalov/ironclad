#!/bin/bash

echo "DOWNLOADING DATA SOURCES...saving files to './storage'"
mkdir -p ironclad/storage

echo " > images and probe"
wget --no-check-certificate \
  "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/jpulido6_jh_edu/IQAhv09T__00RqMeUDkTC8Y4AYiaEwHhDKJtiN16q0o0Ke4?e=b4FzmU&download=1" \
  -O ironclad/storage/multi_image_identities.tar

echo " > Extracting..."
tar -xvf ironclad/storage/multi_image_identities.tar -C ironclad/storage/


rm ironclad/storage/multi_image_identities.tar

echo "DONE."
