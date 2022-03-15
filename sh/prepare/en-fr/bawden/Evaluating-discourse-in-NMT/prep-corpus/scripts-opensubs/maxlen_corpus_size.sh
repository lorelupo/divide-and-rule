


awk '$0=""NF' | awk -v "maxlen=$1" '$0 < maxlen' | wc -l