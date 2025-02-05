import json
import random

with open('test.txt','w') as testf:
    with open('train.txt','w') as trainf:
        with open("Games.inter.json", 'r') as f:
            inters = json.load(f)
            f.close()

            nuser = len(inters)

            user_item = [[] * nuser]
            for user in inters:
                user_id = int(user)
                traj = inters[user]
                for i, item in enumerate(traj):
                    if i == len(traj)-1:
                        testf.writelines('{} {} {}\n'.format(user_id, item, 1))

                        random_num = random.randint(0, 100)
                        # 判断随机数是否小于 5（即 5% 的概率）
                        if random_num < 3:
                            trainf.writelines('{} {} {}\n'.format(user_id, item, 1))
                        else:
                            pass
                    else:
                        trainf.writelines('{} {} {}\n'.format(user_id, item, 1))
        trainf.close()
    testf.close()