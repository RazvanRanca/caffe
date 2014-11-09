
TODO

max batch: train_batch, val_batch (test_iter)
06:
05: 8,8
04:            | 1,10 (80)
03: 5,1
02: 11,2 (322)
01:

Alex:
-> try new inadcl_o solver mode
-> avoid Redbox Perfects
   -> run inadcl_o_noRedPerfs for 1000 on graphic05 to compare
   -> ok, should not avoid them
   
-> train inadcl_o more
   -> lr5 non stop
   -> not good (?)
   -> BIGGER BATCH SIZE
   
-> scrape_o
   -> 1500 lr4
      -> iter1500 67% val
      -> iter1000 64% val
   -> 1000 lr5, 1000 lr6
      ->
   -> IDEAS!
      -> rm scrape good zone bad
      -> rm all zone bad
      
-> misal_o
   -> 200 lr4
      -> iter200 82% val

-> train soil_high_o
   -> 1000 lr4, 1000 lr5
      -> iter2000 83% val
   -> iter3300 (85,81)%

-> train water_high_o
   -> 1000 lr4, 1000 lr5
      -> 
      
-> graphic04 6 GB!!
   -> but compute capability (threads) is bottleneck

-> graphic02 batchsize 10!
   -> hence why soil good perf?
   
-> freeze backprop 
   -> inadcl_o: no, not overfitting
   -> misal_o: why not, smaller set
   
-> LOOK INTO TEST ITER
   -> run on graphic02
   -> val_18.txt vs val_19.txt
   -> with alexnet because light enough
   -> 
-> graphic02: 
-> eliminate barcode images with intersect wrongly classified imgs
   over all classes, and propose this as preprocessing step when we
   go to CP

Raz:
-> pool classifications for multi image queries
-> write preds.txt file - joint name not image name!!
demo flag names exact, say if merged


