#!/bin/bash

sumCoref=0
sumPronominalCoref=0
# for all the files in the current directory
# E.g. /home/getalp/dinarelm/work/tools/CoreNLP/stanford-corenlp-4.2.0/data/iwslt_en-fr/IWSLTtrain_chunks
for file in ./*out; do
    # The following command sum the number of sentences that have at least
    # a reference whose coreferent can be found in the past context.
    (( sumCoref += $(grep -A10000 'Coreference' $file | perl -pe '{s/^\s+//g;}' | grep '^(' | cut -d',' -f1,4 | perl -pe '{s/\,[0-9]+\]\) \-\> / /g; s/[\(\)]//g;}' | awk '{ if($1 != $2){print $1;} }' | sort -u | wc -l) ))
    (( sumPronominalCoref += $(grep -A10000 'Coreference' $file | grep -e "\"I\"" -e "\"me\"" -e "\"my\"" -e "\"he\"" -e "\"him\"" -e "\"she\"" -e "\"her\"" -e "\"it\"" -e "\"we\"" -e "\"they\"" -e "\"them\"" -e "\"his\"" -e "\"hers\"" -e "\"its\"" -e "\"their\"" -e "\"theirs\"" -e "\"himself\"" -e "\"herself\"" -e "\"itself\"" -e "\"themselves\"" -e "\"this\"" -e "\"that\"" -e "\"these\"" -e "\"those\"" -e "\"who\"" -e "\"whom\"" -e "\"which\"" -e "\"whose\"" | perl -pe '{s/^\s+//g;}' | grep '^(' | cut -d',' -f1,4 | perl -pe '{s/\,[0-9]+\]\) \-\> / /g; s/[\(\)]//g;}' | awk '{ if($1 != $2){print $1;} }' | sort -u | wc -l) ))
done

echo "Total number of sentences with at least a coreferent in the past: $sumCoref"
echo "Total number of sentences with at least a pronominal antecedent in the past: $sumPronominalCoref"

# Command details: 
#
#  grep -A10000 'Coreference' chunk1.txt.out | \ # grep with after context of 10000, i.e. all possible after context (ok since all coreferences are at the end of thhe file)
#     perl -pe '{s/^\s+//g;}' | \ # strip spaces and newlines
#     grep '^(' | # strip all lines that do not start with "("
#     cut -d',' -f1,4 | # cut everything after the fourth comma, in each line. E.g. (182,2]) -> (175
#     perl -pe '{s/\,[0-9]+\]\) \-\> / /g; s/[\(\)]//g;}' | \ # keep first and last figure of each line. E.g. 183 175
#     awk '{ if($1 != $2){print $1;} }' | # keep the lines where the two figures are different (inter sentential coreference), print the first figure (later reference).
#     sort -u | # keep only one reference per line
#     wc -l # count

