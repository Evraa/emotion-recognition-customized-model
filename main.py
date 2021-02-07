#global imports
import argparse
import sys
import copy
#local imports
import consts
import preprocess
import train



if __name__ == '__main__':

    consts.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--emotions", nargs="+", default=["happy", "sad"],required=False)   
    parser.add_argument("-es", "--epoch_size",  required=False, default=10)   
    parser.add_argument("-bs", "--batch_size", required=False, default=32)   
    args = parser.parse_args()

    try:
        consts._epochs = int(args.epoch_size)
        consts._batch_size = int(args.batch_size)
    except:
        print ("Error: batch size and epoch size need to be integer values")
        sys.exit(0)

    
    emotions_list = list(consts._all_emotions.values())
    for em in args.emotions:
        em = em.lower()
        if em not in emotions_list:
            print (f"Error: `{em}` not one of the emotions included in the dataset")
            print (f'Dataset emotions are: {consts._all_emotions}')
            sys.exit(0)
        consts._class_names.append(em)
        consts._emotions.append(emotions_list.index(em))
    
    if len(args.emotions) <2:
        print ("Error: we need at least two emotions to start with..")
        print (f'{args.emotions}  is not enough.')
        sys.exit(0)

    print ("\nWe are good to go\n")
    print (f'\tepoch size: {args.epoch_size}\n\tbatch size: {args.batch_size}')
    print (f'\tEmotions picked: {consts._class_names}')

    preprocess.run_preprocess()
    train.run_train()