import random
import csv

def calc_bmi(h, w):
    'BMI(体重kg / 身長m^2)を計算して返す'
    bmi = w / (h / 100) ** 2
    if bmi < 18.5: return 'thin'
    if bmi < 25: return 'normal'
    return 'fat'

cnt = {'thin': 0, 'normal': 0, 'fat': 0}
with open('bmi.csv', 'w') as f:
    csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['height','weight','label'])
    for i in range(20000):
        h = random.randint(120,200)
        w = random.randint(35,80)
        label = calc_bmi(h, w)
        cnt[label] += 1
        csv_writer.writerow([h, w, label])

print(cnt)
