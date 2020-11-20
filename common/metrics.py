def acc_metrics(pre,y,threshold):
    pre = [0 if num<threshold else 1 for num in pre]
    cnt = 0
    positive = 0
    true_positive = 0
    pre_positive = 0
    for i in range(len(pre)):
        if pre[i]==y[i]:
            cnt+=1
        if y[i]==1:
            positive+=1
        if y[i]==1 and pre[i]==1:
            true_positive+=1
        if pre[i]==1:
            pre_positive+=1
            
    return cnt,len(pre),true_positive,positive,pre_positive