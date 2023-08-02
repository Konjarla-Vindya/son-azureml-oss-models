
az version

az extension remove -n ml

az extension add -n ml -y  || {
    echo "az extension add -n ml -y failed..."; exit 1;
}


# Use defaults if not passed by workflow inputs

GROUP1=${GROUP:-"sonata-test-rg"}

LOCATION1=${LOCATION:-"southcentralus"}

WORKSPACE1=${WORKSPACE:-"sonata-test-ws"}

az configure --defaults group=$GROUP1 workspace=$WORKSPACE1 location=$LOCATION1

