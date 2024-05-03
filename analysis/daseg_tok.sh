cat ../exp/tasks/data/ICSI_split_with_da/val/val.txt | sed 's/| / |/g' | tr -d '?.,"' | sed 's/$/\n/g' #| tr -s " " "\n"
