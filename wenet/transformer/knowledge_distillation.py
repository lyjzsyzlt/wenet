import torch
import torch.nn as nn

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.common import IGNORE_ID, add_sos_eos


class KD(nn.Module):
    def __init__(self, configs_student, configs_teacher=None):
        '''

        :param configs_student: student模型的配置文件
        :param configs_teacher: teacher模型的配置文件
        :param method: KD的方法，支持PKD
        '''

        super(KD, self).__init__()
        self.student = init_asr_model(configs_student)
        if configs_teacher is not None:
            self.teacher = init_asr_model(configs_teacher)
            self.teacher.eval()
            for _, p in self.teacher.named_parameters():
                p.requires_grad = False
        else:
            self.teacher = None

        self.kd_config = configs_student['kd_conf']
        self.sos = configs_student['output_dim']
        self.eos = configs_student['output_dim']
        # self.v = torch.nn.ModuleList([torch.nn.Linear(256,1) for _ in range(self.student.decoder.num_blocks)])

    def forward(self, feats, feats_lengths, target, target_lengths, cv=False):
        if cv:
            self.student.eval()
        else:
            self.student.train()
        loss_hard, loss_att, loss_ctc, encoder_out_s, encoder_mask, logits_s, dec_outputs_s = self.student(feats,
                                                                                                           feats_lengths,
                                                                                                           target,
                                                                                                           target_lengths)

        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                loss1, _, _, encoder_out_t, _, logits_t, dec_outputs_t = self.teacher(feats, feats_lengths, target,
                                                                                  target_lengths)

            # PKD
            if self.kd_config['method'] == 'PKD':
                if self.kd_config['encoder']['kd']:
                    loss_pkd_e = self.PKD(encoder_out_s, encoder_out_t, index=[0, 4, 7, 11], mask=encoder_mask)
                    # alpha = self.kd_config['encoder']['alpha']
                    alpha = 0.2
                else:
                    loss_pkd_e = torch.FloatTensor([0.0]).cuda()
                    alpha = 0.0

                if self.kd_config['decoder']['kd']:
                    _, ys_out_pad = add_sos_eos(target, self.sos, self.eos, IGNORE_ID)
                    ignore = ys_out_pad == IGNORE_ID  # (B,)
                    loss_pkd_d = self.PKD(dec_outputs_s, dec_outputs_t, index=[2, 5], mask=ignore.unsqueeze(-2))
                    # beta = self.kd_config['decoder']['beta']
                    beta = 0.2
                else:
                    loss_pkd_d = torch.FloatTensor([0.0]).cuda()
                    beta = 0.0

                loss_pkd = alpha * loss_pkd_e + beta * loss_pkd_d
                loss = loss_hard + loss_pkd
                return loss, loss_hard, loss_pkd

            # Multi-level feature distillation
            elif self.kd_config['method'] == 'MLFD':
                if self.kd_config['encoder']['kd']:
                    index = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11]]
                    # index = [list(range(12)), list(range(4,12)), list(range(8, 12)), [11]]
                    loss_mlfd_e = self.MLFD(encoder_out_s, encoder_out_t, index, encoder_mask, attn_type='dot')
                    alpha = 0.2
                    # 这算的是encoder ctc出来的logits
                    # with torch.no_grad():
                    #     logits = self.teacher.ctc.get_logits(encoder_out_t[-1])
                    # lamb = 0.4
                    # loss_ckd = self.CKD(self.student.ctc.get_logits(encoder_out_s[-1]), logits, encoder_mask.reshape(-1,1)) * lamb

                else:
                    loss_mlfd_e = torch.FloatTensor([0.0]).cuda()
                    alpha = 0.0

                if self.kd_config['decoder']['kd']:
                    _, ys_out_pad = add_sos_eos(target, self.sos, self.eos, IGNORE_ID)
                    ignore = ys_out_pad == IGNORE_ID  # (B,)
                    index = [[0, 1, 2, 3, 4], [5]]
                    loss_mlfd_d = self.MLFD(dec_outputs_s, dec_outputs_t, index, ignore.unsqueeze(-2), attn_type='add')
                    # beta = self.kd_config['decoder']['beta']
                    beta = 0.2
                else:
                    loss_mlfd_d = torch.FloatTensor([0.0]).cuda()
                    beta = 0.0
                loss_mlfd = alpha * loss_mlfd_e + beta * loss_mlfd_d
                loss = loss_hard + loss_mlfd
                return loss, loss_hard, loss_mlfd

        return loss_hard, loss_att, loss_ctc

    def PKD(self, output_s, output_t, index, mask=None):
        """
        Patient KD
        :param output_s: student模型的隐层输出 List[B x T x D, B x T x D, ...]
        :param output_t: teacher模型的隐层输出 List[B x T x D, B x T x D, ...]
        :param index: teacher模型被选择的层id, 从0开始编号, eg: [1,2,3,4]
        :return: PKD loss
        """
        d_s = output_s[0].shape[2]
        d_t = output_t[0].shape[2]
        assert d_s == d_t, "the dims of T and S should be same"
        batch = output_s[0].shape[0]

        output_t = [output_t[i].masked_fill(~mask.transpose(1, 2), 0.0) for i in index]
        output_s = [output_s[i].masked_fill(~mask.transpose(1, 2), 0.0) for i in range(len(index))]

        criterion = nn.MSELoss(reduce=True, size_average=False)
        loss = 0.0
        for i in range(len(index)):
            loss += criterion(
                output_s[i] / torch.norm(output_s[i], p=float('inf'), dim=(1, 2)).unsqueeze(1).unsqueeze(2),
                output_t[i] / torch.norm(output_t[i], p=float('inf'), dim=(1, 2)).unsqueeze(1).unsqueeze(2)) / (
                            batch * len(index))
        return loss

    def MLFD(self, output_s, output_t, index, mask=None, attn_type='dot', p=float('inf')):
        """
        Multi-level feature distillation
        :param output_s: student模型的隐层输出 List[B x T x D, B x T x D, ...]
        :param output_t: teacher模型的隐层输出 List[B x T x D, B x T x D, ...]
        :param mask:
        :param attn_type: ['dot', 'add']
        :return:
        """
        assert output_s[0].shape[2] == output_t[0].shape[2], "the dims of the student and teacher should be same"
        slayer = len(output_s)
        tlayer = len(output_t)
        batch = output_s[0].shape[0]
        T = output_s[0].shape[1]
        D = output_s[0].shape[2]

        output_t = [output_t[i].masked_fill(~mask.transpose(1, 2), 0.0) for i in range(tlayer)]  # enc_states ([batch, Tmax, dm])
        output_t = [output_t[i] / torch.norm(output_t[i], p=p, dim=(1, 2)).unsqueeze(1).unsqueeze(2) for i in range(tlayer)]
        output_t = torch.cat(output_t, dim=0)

        output_s = [output_s[i].masked_fill(~mask.transpose(1, 2), 0.0) for i in
                    range(slayer)]  # enc_states ([batch, Tmax, dm])
        output_s = [output_s[i] / torch.norm(output_s[i], p=p, dim=(1, 2)).unsqueeze(1).unsqueeze(2) for i in
                    range(slayer)]
        output_s = torch.cat(output_s, dim=0)

        if attn_type == 'dot':
            s = output_s.view(slayer, -1, T, D).transpose(0, 1).transpose(1, 2)
            t = output_t.view(tlayer, -1, T, D).transpose(0, 1).transpose(1, 2)

            # 全局attention
            # score = torch.matmul(s, t.transpose(-2, -1))
            # score = torch.softmax(score, dim=-1)
            # output_t = torch.matmul(score, t).transpose(0, 2).transpose(1, 2).reshape(slayer, -1, T, D)

            # 分块进行attention
            res=[]
            for i in range(slayer):
                s=output_s.view(slayer, -1, T, D)[i].unsqueeze(0).transpose(0, 1).transpose(1, 2)
                t=output_t.view(tlayer, -1, T, D)[index[i]].transpose(0, 1).transpose(1, 2)
                score = torch.matmul(s, t.transpose(-2, -1))
                score = torch.softmax(score, dim=-1)
                res.append(torch.matmul(score, t).transpose(0, 2).transpose(1, 2).reshape(-1, T, D))
            output_t = torch.cat(res, dim=0).view(slayer, -1, T, D)

            criterion = nn.MSELoss(reduce=False, size_average=False)
            loss = 0.0
            output_s = output_s.view(slayer, -1, T, D)
            # print(self.teacher._calc_att_loss(x[2], mask, target, target_lengths)[0])
            for i in range(slayer):
                loss += criterion(
                    output_s[i] / torch.norm(output_s[i], p=p, dim=(1, 2)).unsqueeze(1).unsqueeze(2),
                    output_t[i] / torch.norm(output_t[i], p=p, dim=(1, 2)).unsqueeze(1).unsqueeze(
                        2)).masked_fill(~mask.transpose(1, 2), 0.0).sum() / (batch * slayer)
            return loss
        elif attn_type == 'add':
            res=[]
            for i in range(slayer):
                if len(index[i])==1:
                    s = output_s.view(slayer, -1, T, D)[i].unsqueeze(0).unsqueeze(1).repeat(1, len(index[i]), 1, 1, 1)
                    t = output_t.view(tlayer, -1, T, D)[index[i]].unsqueeze(0).unsqueeze(0)
                    score = self.v[i](s + t)
                    score = torch.softmax(score, dim=1)
                    t = output_t.view(tlayer,-1,T,D)[index[i]].unsqueeze(0).unsqueeze(0)
                    res.append(torch.sum(score*t,dim=1).view(-1, T, D))
                else:
                    s = output_s.view(slayer, -1, T, D)[i].unsqueeze(0).unsqueeze(1).repeat(1, len(index[i]), 1, 1, 1)
                    t = output_t.view(tlayer, -1, T, D)[index[i]].unsqueeze(0)
                    score = self.v[i](s + t)
                    score = torch.softmax(score, dim=1)
                    t = output_t.view(tlayer, -1, T, D)[index[i]].unsqueeze(0)
                    res.append(torch.sum(score*t,dim=1).view(-1, T, D))
            output_t = torch.cat(res, dim=0).view(slayer, -1, T, D)

            criterion = nn.MSELoss(reduce=False, size_average=False)
            loss = 0.0
            output_s = output_s.view(slayer, -1, T, D)
            # print(self.teacher._calc_att_loss(x[2], mask, target, target_lengths)[0])
            for i in range(slayer):
                loss += criterion(
                    output_s[i] / torch.norm(output_s[i], p=p, dim=(1, 2)).unsqueeze(1).unsqueeze(2),
                    output_t[i] / torch.norm(output_t[i], p=p, dim=(1, 2)).unsqueeze(1).unsqueeze(
                        2)).masked_fill(~mask.transpose(1, 2), 0.0).sum() / (batch * slayer)
            return loss

    def CKD(self, logits_s, logits_t, mask):
        """
        最原始的KD算法. Hinton 2015
        :param logits_s: student模型的logits B x T x Vocab_size
        :param logits_t: teacher模型的logits  B x T x Vocab_size
        :param mask: targets padding对应的mask. (B x T) x 1
        :return:
        """
        criterion = nn.KLDivLoss(reduction="none")
        batch = logits_s.shape[0]
        vocab_size = logits_s.shape[2]
        logits_s = logits_s.view(-1, vocab_size) # (B x T) x Vocab_size
        logits_t = logits_t.view(-1, vocab_size) # (B x T) x Vocab_size

        loss = criterion(torch.log_softmax(logits_s, dim=1), logits_t)
        loss = loss.masked_fill(mask, 0).sum()/batch
        return loss



