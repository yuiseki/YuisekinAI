LINES=`wc -l tmp/text.txt | awk '{print $1}'`
echo $LINES

TRAIN_DATA_LINES=$(($LINES*95/100))
TEST_DATA_LINES=$(($LINES-$TRAIN_DATA_LINES))
echo $TRAIN_DATA_LINES
echo $TEST_DATA_LINES

head -n $TRAIN_DATA_LINES tmp/text.txt > tmp/train.txt
tail -n $TEST_DATA_LINES tmp/text.txt > tmp/test.txt
