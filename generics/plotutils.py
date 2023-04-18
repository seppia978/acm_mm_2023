import torch
import matplotlib.pyplot as plt
import os

def plot_pairs(a, b, idx, layer, idx2=-1, DeT=None, path=None,inc=10,alpha=None,th=1):
    savep = '/homes/spoppi/pycharm_projects/inspecting_twin_models/vgg16/f29'

    if DeT is not None:
        a, b = DeT(a), DeT(b)

    fig, (a1,a2) = plt.subplots(1,2)
    if idx2 == -1:
        inc_=inc
        r=(idx,min(idx+inc_, len(a)))
        dist = 0
        for i in range(*r):
            if alpha:
                if alpha[i] <= th:
                    fig.suptitle(f'{layer=} - {i=} --{round(float(alpha[i].detach().cpu().numpy()),3)} -->> {torch.sqrt((torch.pow((a[i]-b[i]),2)).sum())}')
                    a1.imshow(a[i].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
                    a1.set_title('Standard')
                    a1.axes.get_xaxis().set_visible(False)
                    a1.axes.get_yaxis().set_visible(False)

                    a2.imshow(b[i].detach().cpu().permute(1,2,0).numpy(), cmap='gray')
                    a2.set_title('Modified')
                    a2.axes.get_xaxis().set_visible(False)
                    a2.axes.get_yaxis().set_visible(False)
                    if i < r[-1] - 1:
                        fig.show()
            else:
                if torch.sqrt((torch.pow((a[i] - b[i]), 2)).sum()) > dist:
                    dist = torch.sqrt((torch.pow((a[i] - b[i]), 2)).sum())
                    print(f'{i=}\t{dist}')

                fig.suptitle(
                    f'{layer=} - {i=} ---->> {dist}')
                A = a[i].clone()
                A-=A.min()
                A/=A.max()
                a1.imshow(A.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
                a1.set_title('Standard')
                a1.axes.get_xaxis().set_visible(False)
                a1.axes.get_yaxis().set_visible(False)

                B = b[i].clone()
                B-=B.min()
                B/=B.max()
                a2.imshow(B.detach().cpu().permute(1, 2, 0).numpy(), cmap='gray')
                a2.set_title('Modified')
                a2.axes.get_xaxis().set_visible(False)
                a2.axes.get_yaxis().set_visible(False)
                if i < r[-1] - 1:
                    fig.show()
    elif idx2 == -2:
        r = (0,10)
        for i in range(*r):
            fig.suptitle(f'{layer=} - {idx=} - {i=} ----- {torch.sqrt((torch.pow((a[idx, i] - b[idx, i]), 2)).sum())}')
            a1.imshow(a[idx, i].detach().cpu().numpy(), cmap='gray')
            a1.set_title('Standard')
            a1.axes.get_xaxis().set_visible(False)
            a1.axes.get_yaxis().set_visible(False)

            a2.imshow(b[idx, i].detach().cpu().numpy(), cmap='gray')
            a2.set_title('Modified')
            a2.axes.get_xaxis().set_visible(False)
            a2.axes.get_yaxis().set_visible(False)

            if path is not None:
                if not os.path.isdir(os.path.join(path,str(idx))):
                    os.mkdir(os.path.join(path,str(idx)))
                fig.savefig(os.path.join(path,str(idx),f'{i}.png'))
            if i < r[-1] - 1:
                fig.show()

    elif idx2 >= 0:
        inc_ = inc
        r = (idx, min(idx + inc_, len(a)))
        for i in range(*r):
            if alpha[i] <= th:
                fig.suptitle(
                    f'{layer=} - {i=} --{round(float(alpha[i].detach().cpu().numpy()), 3)} -->> {torch.sqrt((torch.pow((a[i,idx2] - b[i,idx2]), 2)).sum())}'
                )
                a1.imshow(a[i,idx2].detach().cpu().numpy(),cmap='gray')
                a1.set_title('Standard')
                a1.axes.get_xaxis().set_visible(False)
                a1.axes.get_yaxis().set_visible(False)

                a2.imshow(b[i,idx2].detach().cpu().numpy(),cmap='gray')
                a2.set_title('Modified')
                a2.axes.get_xaxis().set_visible(False)
                a2.axes.get_yaxis().set_visible(False)
                if i < r[-1] - 1:
                    fig.show()
    try:
        fig.show()
    except:
        pass