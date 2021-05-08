import time

import six
import numpy as np
import torch

class CTCPrefixScore(object):
    """Compute CTC label sequence scores

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, x, blank, eos, xp):
        '''

        :param x: encoder output, L x V
        :param blank:
        :param eos:
        :param xp:
        '''
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = x.shape[0]
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state

        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        # r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r = torch.full((self.input_length, 2), self.logzero, dtype=torch.float32).cuda()
        r[0,1] = self.x[0,self.blank]

        for i in six.moves.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = torch.zeros((self.input_length, 2, len(cs)), dtype=torch.float32).cuda()
        xs = self.x[:, cs]  # 从
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = torch.logaddexp(
            r_prev[:, 0], r_prev[:, 1]
        )  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = torch.zeros((self.input_length, len(cs)), dtype=torch.float32).cuda()
            for i in six.moves.range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum
        # print(r[0][0].shape, log_phi.shape)
        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilites log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        t1=time.time()
        for t in six.moves.range(start, self.input_length):
            # print(r.shape, log_phi.shape)
            r[t, 0] = torch.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = (
                torch.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
            )
            log_psi = torch.logaddexp(log_psi, log_phi[t - 1] + xs[t])
        t2=time.time()
        # get P(...eos|X) that ends with the prefix itself
        # eos_pos = np.where(cs == self.eos)[0]
        eos_pos = torch.where(cs == self.eos)[0]  # eg. tensor([0, 5])
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = torch.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        # print(torch.from_numpy(np.rollaxis(r.detach().cpu().numpy(), 2)).shape, r.transpose(0,2).shape, r.shape)
        # return log_psi, torch.from_numpy(np.rollaxis(r.detach().cpu().numpy(), 2)).cuda()
        return log_psi, r.transpose(0,2).transpose(1,2), t2-t1
        # return log_psi, self.xp.rollaxis(r, 2)
