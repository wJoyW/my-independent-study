# Malware Classifier

## 使用方法
  1. cmd
  2. python ./malware_classifier.py INPUT.txt
  3. output 'MAL = True' or 'MAL = False'

## 備註
  1. INPUT.txt
    * 一個檔案內僅能一支process
    * 開頭須為[SCAN_START]那行
  2. cmd 指令時input data的路徑不能有空白
  3. 執行時，rf.pkl和feature.pkl要跟malware_classifier.py在同個資料夾內