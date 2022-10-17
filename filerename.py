import os


# filelist1 = os.listdir('./wav')
# for i in range(len(filelist1)):
#     os.rename('./wav/%s'%filelist1[i],'./wav/kword-org-%s.wav'%i)

filelist2 = os.listdir('./bg')
for i in range(len(filelist2)):
    os.rename('./bg/%s'%filelist2[i],'./bg/bg-org-%s.wav'%i)