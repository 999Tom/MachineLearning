import csv
with open('./anonymisedData/studentVle.csv', newline='') as csvfile:

  # 以冒號分隔欄位，讀取檔案內容
  rows = csv.reader(csvfile, delimiter=':')
  thesumA = 0
  thesumB = 0
  thesumC = 0
  thesumD = 0
  for row in rows:
    looking_row = row[0].split(",")
    #print(looking_row)
    if(looking_row[2] == '"631441"'):
      thesumA = thesumA + int(looking_row[-1].replace('"',''))
    if(looking_row[2] == '"635988"'):
      thesumB = thesumB + int(looking_row[-1].replace('"',''))
    if(looking_row[2] == '"358341"'):
      thesumC = thesumC + int(looking_row[-1].replace('"',''))
    if(looking_row[2] == '"2679821"'):
      thesumD = thesumD + int(looking_row[-1].replace('"',''))
  print(thesumA)
  print(thesumB)
  print(thesumC)
  print(thesumD)
