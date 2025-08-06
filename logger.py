import datetime

tstamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logfile = f'{tstamp}_log.txt'

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text + '\n')
