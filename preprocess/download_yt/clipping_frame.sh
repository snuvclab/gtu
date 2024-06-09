#!/bin/bash
set -e
VID_DIR=$1
PJ_DIRS=$2


file_with_extension=$(basename "$VID_DIR")
file_name="${file_with_extension%.*}"


mkdir -p ${PJ_DIRS}
PJ_DIRS=${PJ_DIRS}/${file_name}
mkdir -p ${PJ_DIRS}/images



if [ $# -ge 4 ]; then
    echo "Parse from ${3} to ${4}"
    start_time="$3"
    # Extract minutes and seconds using the cut command
    minutes=$(echo $start_time | cut -d':' -f1)
    seconds=$(echo $start_time | cut -d':' -f2)

    # Convert minutes to seconds and add to seconds
    start_seconds=$((10#$minutes * 60 + 10#$seconds))

    # -----------------------------------------------------------
    input_time="$4"
    # Extract minutes and seconds using the cut command
    minutes=$(echo $input_time | cut -d':' -f1)
    seconds=$(echo $input_time | cut -d':' -f2)

    # Convert minutes to seconds and add to seconds
    end_seconds=$((minutes * 60 + seconds))

    echo ffmpeg -i ${VID_DIR} -q:v 4 -ss ${start_seconds} -to ${end_seconds} $PJ_DIRS/images/"%06d.png"
    ffmpeg -i ${VID_DIR} -q:v 4 -ss ${start_seconds} -to ${end_seconds} $PJ_DIRS/images/"%06d.png" 

else

    ffmpeg -i ${VID_DIR} -q:v 4 $PJ_DIRS/images/"%05d.png"
fi


echo "Finished Processing videos in ${PJ_DIRS}"