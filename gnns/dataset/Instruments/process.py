import json

with open('test.txt','w') as testf:
    with open('train.txt','w') as trainf:
        with open("Instruments.inter.json", 'r') as f:
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
                    else:
                        trainf.writelines('{} {} {}\n'.format(user_id, item, 1))
        trainf.close()
    testf.close()