#global imports
import argparse
import sys
#local imports
import consts



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--emotions", nargs="+", default=["happy", "sad"],required=False)   
    parser.add_argument("-es", "--epoch_size",  required=False, default=10)   
    parser.add_argument("-bs", "--batch_size", required=False, default=32)   
    args = parser.parse_args()

    if not isinstance(args.batch_size, int) or\
        not isinstance(args.epoch_size, int):
        print ("Error: batch size and epoch size need to be integer values")
        sys.exit(0)

    emotions = []
    for em in args.emotions:
        em = em.lower()
        if em not in consts._all_emotions.values():
            print (f"Error: `{em}` not one of the emotions included in the dataset")
            print (f'Dataset emotions are: {consts._all_emotions}')
            sys.exit(0)
        emotions.append(em)
    
    if len(args.emotions) <2:
        print ("Error: we need at least two emotions to start with..")
        print (f'{args.emotions}  is not enough.')
        sys.exit(0)

    print ("\nWe are good to go\n")
    print (f'\tepoch size: {args.epoch_size}\n\tbatch size: {args.batch_size}')
    print (f'\tEmotions picked: {emotions}')

    