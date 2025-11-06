from ot.lp import emd
from ot.utils import list_to_array
from ot.backend import get_backend
from ot.bregman import sinkhorn
from scipy.spatial.distance import cdist
import warnings
import torch
import numpy as np

from ..metrics import eva_foscttm_ot

def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mimics scipy.spatial.distance.correlation using PyTorch.
    Computes 1 - Pearson correlation between rows of x and y.
    """
    # Center the inputs
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)

    # Normalize by L2 norm (standard deviation)
    x_norm = x_centered / x_centered.norm(dim=1, keepdim=True)
    y_norm = y_centered / y_centered.norm(dim=1, keepdim=True)

    # Cosine similarity of normalized centered vectors = Pearson correlation
    correlation = torch.matmul(x_norm, y_norm.T)

    # Correlation distance = 1 - correlation
    return 1 - correlation


def coot(
        X,
        Y,
        prior=None,
        wx_samp=None,
        wx_feat=None,
        wy_samp=None,
        wy_feat=None,
        epsilon=0,
        alpha=0,
        M_samp=None,
        M_feat=None,
        warmstart=None,
        nits_bcd=100,
        tol_bcd=1e-7,
        eval_bcd=1,
        nits_ot=500,
        tol_sinkhorn=1e-7,
        early_stopping_tol=1e-6,
        log=False,
        verbose=False,
):
    # Main function

    X, Y = list_to_array(X, Y)
    # backend infer
    nx = get_backend(X, Y)

    if isinstance(epsilon, float) or isinstance(epsilon, int):
        eps_samp, eps_feat = epsilon, epsilon
    else:
        if len(epsilon) != 2:
            raise ValueError(
                "Epsilon must be either a scalar or an indexable object of length 2."
            )
        else:
            eps_samp, eps_feat = epsilon[0], epsilon[1]

    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha_samp, alpha_feat = alpha, alpha
    else:
        if len(alpha) != 2:
            raise ValueError(
                "Alpha must be either a scalar or an indexable object of length 2."
            )
        else:
            alpha_samp, alpha_feat = alpha[0], alpha[1]

    # constant input variables
    if M_samp is None or alpha_samp == 0:
        M_samp, alpha_samp = 0, 0
    if M_feat is None or alpha_feat == 0:
        M_feat, alpha_feat = 0, 0

    nx_samp, nx_feat = X.shape
    ny_samp, ny_feat = Y.shape

    # measures on rows and columns
    if wx_samp is None:
        wx_samp = nx.ones(nx_samp, type_as=X) / nx_samp
    if wx_feat is None:
        wx_feat = nx.ones(nx_feat, type_as=X) / nx_feat
    if wy_samp is None:
        wy_samp = nx.ones(ny_samp, type_as=Y) / ny_samp
    if wy_feat is None:
        wy_feat = nx.ones(ny_feat, type_as=Y) / ny_feat

    wxy_samp = wx_samp[:, None] * wy_samp[None, :]
    wxy_feat = wx_feat[:, None] * wy_feat[None, :]

    # initialize coupling and dual vectors
    if warmstart is None:
        pi_samp, pi_feat = (
            wxy_samp,
            wxy_feat,
        )  # shape nx_samp x ny_samp and nx_feat x ny_feat
        duals_samp = (
            nx.zeros(nx_samp, type_as=X),
            nx.zeros(ny_samp, type_as=Y),
        )  # shape nx_samp, ny_samp
        duals_feat = (
            nx.zeros(nx_feat, type_as=X),
            nx.zeros(ny_feat, type_as=Y),
        )  # shape nx_feat, ny_feat
    else:
        pi_samp, pi_feat = warmstart["pi_sample"], warmstart["pi_feature"]
        duals_samp, duals_feat = warmstart["duals_sample"], warmstart["duals_feature"]

    if prior is not None:
        # prior sample is the plan obtained from OT
        # prior feature is the initial adjcaent matrix
        pi_samp, pi_feat = prior["prior_samp"], prior["prior_feat"]

    # pi_feat = prior_feat
    # initialize log
    list_coot = [float("inf")]
    err = tol_bcd + 1e-3

    for idx in range(nits_bcd):
        pi_samp_prev = nx.copy(pi_samp)
        # update sample coupling
        # ot_cost = cdist(X, Y @ pi_feat.T, metric="correlation")
        ot_cost = cdist(X, Y @ pi_feat.T, metric="cosine")
        if eps_samp > 0 and prior is not None:
            pi_samp, dict_log = sinkhorn_knopp_with_prior(
                a=wx_samp,
                b=wy_samp,
                M=ot_cost,
                reg=eps_samp,
                prior=pi_samp,
                methods="samp",
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True, 
                warmstart=duals_samp,
            )
            duals_samp = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
            coot_samp = nx.sum(ot_cost * pi_samp)
            # print(evaluate(pi_samp))
        elif eps_samp > 0 and prior is None:
            pi_samp, dict_log = sinkhorn(
                a=wx_samp,
                b=wy_samp,
                M=ot_cost,
                reg=eps_samp,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_samp,
            )
            duals_samp = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))

        else:
            warnings.warn("Epsilon is needed for a continuous sanmple-level transport plan. ")
            return

        # update feature coupling
        A_squeezed = nx.squeeze(X.T)
        B_squeezed = nx.squeeze(Y.T @ pi_samp.T)
        jitter = 1e-10
        A_final = A_squeezed + np.random.randn(*A_squeezed.shape) * jitter
        B_final = B_squeezed + np.random.randn(*B_squeezed.shape) * jitter
        
        # ot_cost = cdist(A_final, B_final, metric="correlation")
        ot_cost = cdist(A_final, B_final, metric="cosine")
        if eps_feat > 0 and prior is not None:
            pi_feat, dict_log = sinkhorn_knopp_with_prior(
                a=wx_feat,
                b=wy_feat,
                M=ot_cost,
                reg=eps_feat,
                prior=pi_feat,
                methods="feat", 
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_feat,
            )
            duals_feat = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
            # print("the sum of the feature OT plan at first row is: " + str(pi_feat[0].sum()))
            # print("the local information preservation is: " + str(pi_feat[0][0:4].sum() / (pi_feat[0].sum())))
            coot_feat = nx.sum(ot_cost * pi_feat)
        elif eps_feat > 0 and prior is None:
            pi_feat, dict_log = sinkhorn(
                a=wx_feat,
                b=wy_feat,
                M=ot_cost,
                reg=eps_feat,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_feat,
            )
            duals_feat = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
        else:
            warnings.warn("Epsilon is needed for a continuous feature-level transport plan. ")
            return

        if idx % eval_bcd == 0:
            # update error
            err = nx.sum(nx.abs(pi_samp - pi_samp_prev))

            # COOT part
            # Why minizing the overall cost on feature?
            # Try with sample,
            # coot = nx.sum(ot_cost * pi_feat)
            if alpha_samp != 0:
                coot = coot + alpha_samp * nx.sum(M_samp * pi_samp)
            # Entropic part
            # Also needs modify, although do not influence the general results, just for loss printing purpose.
            # if eps_samp != 0:
            #     coot = coot + eps_samp * nx.kl_div(pi_samp, prior_samp)
            # if eps_feat != 0:
            #     coot = coot + eps_feat * nx.kl_div(pi_feat, prior_feat)
            list_coot.append(coot_samp)

            if err < tol_bcd or abs(list_coot[-2] - list_coot[-1]) < early_stopping_tol:
                break

            if verbose:
                print(
                    "CO-Optimal Transport cost at iteration {}: sample-level OT cost:{}, feature-level OT cost:{}".format(
                        idx + 1, coot_samp, coot_feat
                    )
                )

    # sanity check
    if nx.sum(nx.isnan(pi_samp)) > 0 or nx.sum(nx.isnan(pi_feat)) > 0:
        warnings.warn("There is NaN in coupling.")

    if log:
        dict_log = {
            "duals_sample": duals_samp,
            "duals_feature": duals_feat,
            "distances": list_coot[1:],
        }

        return pi_samp, pi_feat, dict_log

    else:
        return pi_samp, pi_feat





def sinkhorn_knopp_with_prior(
    a,
    b,
    M,
    reg,
    prior,
    methods,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    warmstart=None, 
    **kwargs,
):
    a, b, M = list_to_array(a, b, M)

    nx = get_backend(M, a, b)

    if len(a) == 0:
        a = nx.full((M.shape[0],), 1.0 / M.shape[0], type_as=M)
    if len(b) == 0:
        b = nx.full((M.shape[1],), 1.0 / M.shape[1], type_as=M)

    # init data
    dim_a = len(a)
    dim_b = b.shape[0]

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, n_hists), type_as=M) / dim_a
            v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        else:
            u = nx.ones(dim_a, type_as=M) / dim_a
            v = nx.ones(dim_b, type_as=M) / dim_b
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    # prior = nx.maximum(prior, 1e-100)
    # if methods == 'samp':
    #     K = prior * nx.exp(M / (-reg))
    # elif methods == 'feat':
    #     prior = prior + 1e-5
    #     K = prior * nx.exp(M / (-reg))
    # K = prior * nx.exp(M / (-reg))
    prior = nx.maximum(prior, 1e-10)
    K = prior * nx.exp(M / (-reg))
    
    Kp = (1 / a).reshape(-1, 1) * K
    err = 1
    for ii in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = nx.dot(K.T, u)
        v = b / KtransposeU
        u = 1.0 / nx.dot(Kp, v)
        if (
            nx.any(KtransposeU == 0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Warning: numerical errors at iteration %d" % ii)
            u = uprev
            v = vprev
            break
        if ii % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                tmp2 = nx.einsum("ik,ij,jk->jk", u, K, v)
            else:
                # compute right marginal tmp2= (diag(u)Kdiag(v))^T1
                tmp2 = nx.einsum("i,ij,j->j", u, K, v)
            err = nx.norm(tmp2 - b)  # violation of marginal
            if log:
                log["err"].append(err)

            if err < stopThr:
                break
            if verbose:
                if ii % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))
    else:
        if warn:
            warnings.warn(
                "Sinkhorn did not converge. You might want to "
                "increase the number of iterations `numItermax` "
                "or the regularization parameter `reg`."
            )
    if log:
        log["niter"] = ii
        log["u"] = u
        log["v"] = v

    if n_hists:  # return only loss
        res = nx.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix
        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))





def coot_emb(
        X, #scRNA cell embeddings
        Y, # scATAC cell embeddings
        G, # gene embeddings
        P, # peak embeddings
        prior=None,
        wx_samp=None,
        wx_feat=None,
        wy_samp=None,
        wy_feat=None,
        epsilon=0,
        alpha=0,
        M_samp=None,
        M_feat=None,
        warmstart=None,
        nits_bcd=100,
        tol_bcd=1e-7,
        eval_bcd=1,
        nits_ot=500,
        tol_sinkhorn=1e-7,
        early_stopping_tol=1e-6,
        log=False,
        verbose=False,
):
    # Main function
    # For distance calculation,
    # Given scRNA-seq cell embedding, scATAC-seq cell embedding, gene embedding, peak embedding
    # Given feature correspondence matrix, Cell Distance Matrix defined as
    # {feature correspondence matrix @ ｜RNA cell embedding * gene embedding - ATAC cell embedding * peak embedding} for one single cell pair
    ## TODO: Still, correlation-based distance could be better.
    # Given cell correspondence matrix, Feature Distance Matrix defined as
    # {cell correspondence matrix @ |gene embedding * RNA cell embedding - peak embedding * ATAC cell embedding|} for one single freature pair

    X, Y, G, P = list_to_array(X, Y, G, P)
    # backend infer
    nx = get_backend(X, Y, G, P)

    if isinstance(epsilon, float) or isinstance(epsilon, int):
        eps_samp, eps_feat = epsilon, epsilon
    else:
        if len(epsilon) != 2:
            raise ValueError(
                "Epsilon must be either a scalar or an indexable object of length 2."
            )
        else:
            eps_samp, eps_feat = epsilon[0], epsilon[1]

    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha_samp, alpha_feat = alpha, alpha
    else:
        if len(alpha) != 2:
            raise ValueError(
                "Alpha must be either a scalar or an indexable object of length 2."
            )
        else:
            alpha_samp, alpha_feat = alpha[0], alpha[1]

    # constant input variables
    if M_samp is None or alpha_samp == 0:
        M_samp, alpha_samp = 0, 0
    if M_feat is None or alpha_feat == 0:
        M_feat, alpha_feat = 0, 0

    nx_samp = X.shape[0]
    nx_feat = G.shape[0]
    ny_samp = Y.shape[0]
    ny_feat = P.shape[0]

    # measures on rows and columns
    if wx_samp is None:
        wx_samp = nx.ones(nx_samp, type_as=X) / nx_samp
    if wx_feat is None:
        wx_feat = nx.ones(nx_feat, type_as=X) / nx_feat
    if wy_samp is None:
        wy_samp = nx.ones(ny_samp, type_as=Y) / ny_samp
    if wy_feat is None:
        wy_feat = nx.ones(ny_feat, type_as=Y) / ny_feat

    wxy_samp = wx_samp[:, None] * wy_samp[None, :]
    wxy_feat = wx_feat[:, None] * wy_feat[None, :]

    # initialize coupling and dual vectors
    if warmstart is None:
        pi_samp, pi_feat = (
            wxy_samp,
            wxy_feat,
        )  # shape nx_samp x ny_samp and nx_feat x ny_feat
        duals_samp = (
            nx.zeros(nx_samp, type_as=X),
            nx.zeros(ny_samp, type_as=Y),
        )  # shape nx_samp, ny_samp
        duals_feat = (
            nx.zeros(nx_feat, type_as=X),
            nx.zeros(ny_feat, type_as=Y),
        )  # shape nx_feat, ny_feat
    else:
        pi_samp, pi_feat = warmstart["pi_sample"], warmstart["pi_feature"]
        duals_samp, duals_feat = warmstart["duals_sample"], warmstart["duals_feature"]

    if prior is not None:
        # prior sample is the plan obtained from OT
        # prior feature is the initial adjcaent matrix
        pi_samp, pi_feat = prior["prior_samp"], prior["prior_feat"]

    # pi_feat = prior_feat
    # initialize log
    list_coot = [float("inf")]
    err = tol_bcd + 1e-3

    for idx in range(nits_bcd):
        pi_samp_prev = nx.copy(pi_samp)
        # ot_cost = cdist(X, Y @ pi_feat.T, metric="correlation")
        # Approach 1:
        # ot_cost = cdist(X @ G.T, Y @ P.T @ pi_feat.T, metric="correlation")
        G_fix = G.clone().detach()
        P_fix = P.clone().detach()
        ot_cost = dist(X @ G_fix.T, Y @ P_fix.T @ pi_feat.T)
        # Approach 2:

        if eps_samp > 0 and prior is not None:
            pi_samp, dict_log = sinkhorn_knopp_with_prior(
                a=wx_samp,
                b=wy_samp,
                M=ot_cost,
                reg=eps_samp,
                prior=pi_samp,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_samp,
            )
            duals_samp = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
            coot_samp = nx.sum(ot_cost * pi_samp)
            # print(evaluate(pi_samp))
        elif eps_samp > 0 and prior is None:
            pi_samp, dict_log = sinkhorn(
                a=wx_samp,
                b=wy_samp,
                M=ot_cost,
                reg=eps_samp,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_samp,
            )
            duals_samp = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))

        else:
            warnings.warn("Epsilon is needed for a continuous sanmple-level transport plan. ")
            return

        # update feature coupling
        # ot_cost = cdist(X.T, Y.T @ pi_samp.T, metric="correlation")

        # ot_cost = cdist(G @ X, P @ Y @ pi_samp.T, metric="correlation")
        X_fix = X.clone().detach()
        Y_fix = Y.clone().detach()
        ot_cost = dist(G @ X_fix.T, P @ Y_fix.T @ pi_samp.T)
        if eps_feat > 0 and prior is not None:
            pi_feat, dict_log = sinkhorn_knopp_with_prior(
                a=wx_feat,
                b=wy_feat,
                M=ot_cost,
                reg=eps_feat,
                prior=pi_feat,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_feat,
            )
            duals_feat = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
            # print("the sum of the feature OT plan at first row is: " + str(pi_feat[0].sum()))
            # print("the local information preservation is: " + str(pi_feat[0][0:4].sum() / (pi_feat[0].sum())))
            coot_feat = nx.sum(ot_cost * pi_feat)
        elif eps_feat > 0 and prior is None:
            pi_feat, dict_log = sinkhorn(
                a=wx_feat,
                b=wy_feat,
                M=ot_cost,
                reg=eps_feat,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_feat,
            )
            duals_feat = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
        else:
            warnings.warn("Epsilon is needed for a continuous feature-level transport plan. ")
            return

        if idx % eval_bcd == 0:
            # update error
            err = nx.sum(nx.abs(pi_samp - pi_samp_prev))

            # COOT part
            # Why minizing the overall cost on feature?
            # Try with sample,
            # coot = nx.sum(ot_cost * pi_feat)
            if alpha_samp != 0:
                coot = coot + alpha_samp * nx.sum(M_samp * pi_samp)
            # Entropic part
            # Also needs modify, although do not influence the general results, just for loss printing purpose.
            # if eps_samp != 0:
            #     coot = coot + eps_samp * nx.kl_div(pi_samp, prior_samp)
            # if eps_feat != 0:
            #     coot = coot + eps_feat * nx.kl_div(pi_feat, prior_feat)
            list_coot.append(coot_samp)

            if err < tol_bcd or abs(list_coot[-2] - list_coot[-1]) < early_stopping_tol:
                break

            if verbose:
                print(
                    "CO-Optimal Transport cost at iteration {}: sample-level OT cost:{}, feature-level OT cost:{}".format(
                        idx + 1, coot_samp, coot_feat
                    )
                )
    print("FOSCTTM score via Co-OT: " + str(eva_foscttm_ot(pi_samp, X.shape[0])))
    # sanity check
    if nx.sum(nx.isnan(pi_samp)) > 0 or nx.sum(nx.isnan(pi_feat)) > 0:
        warnings.warn("There is NaN in coupling.")

    if log:
        dict_log = {
            "duals_sample": duals_samp,
            "duals_feature": duals_feat,
            "distances": list_coot[1:],
        }

        return pi_samp, pi_feat, dict_log

    else:
        return pi_samp, pi_feat



def ucoot(
        X,
        Y,
        prior=None,
        wx_samp=None,
        wx_feat=None,
        wy_samp=None,
        wy_feat=None,
        epsilon=0,
        reg_m = 1,
        alpha=0,
        M_samp=None,
        M_feat=None,
        warmstart=None,
        nits_bcd=100,
        tol_bcd=1e-7,
        eval_bcd=1,
        nits_ot=500,
        tol_sinkhorn=1e-7,
        early_stopping_tol=1e-6,
        log=False,
        verbose=False,
):
    # Main function

    X, Y = list_to_array(X, Y)
    # backend infer
    nx = get_backend(X, Y)

    if isinstance(epsilon, float) or isinstance(epsilon, int):
        reg_samp, reg_feat = reg_m, reg_m
    
    if isinstance(epsilon, float) or isinstance(epsilon, int):
        eps_samp, eps_feat = epsilon, epsilon
    else:
        if len(epsilon) != 2:
            raise ValueError(
                "Epsilon must be either a scalar or an indexable object of length 2."
            )
        else:
            eps_samp, eps_feat = epsilon[0], epsilon[1]

    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha_samp, alpha_feat = alpha, alpha
    else:
        if len(alpha) != 2:
            raise ValueError(
                "Alpha must be either a scalar or an indexable object of length 2."
            )
        else:
            alpha_samp, alpha_feat = alpha[0], alpha[1]

    # constant input variables
    if M_samp is None or alpha_samp == 0:
        M_samp, alpha_samp = 0, 0
    if M_feat is None or alpha_feat == 0:
        M_feat, alpha_feat = 0, 0

    nx_samp, nx_feat = X.shape
    ny_samp, ny_feat = Y.shape

    # measures on rows and columns
    if wx_samp is None:
        wx_samp = nx.ones(nx_samp, type_as=X) / nx_samp
    if wx_feat is None:
        wx_feat = nx.ones(nx_feat, type_as=X) / nx_feat
    if wy_samp is None:
        wy_samp = nx.ones(ny_samp, type_as=Y) / ny_samp
    if wy_feat is None:
        wy_feat = nx.ones(ny_feat, type_as=Y) / ny_feat

    wxy_samp = wx_samp[:, None] * wy_samp[None, :]
    wxy_feat = wx_feat[:, None] * wy_feat[None, :]

    # initialize coupling and dual vectors
    if warmstart is None:
        pi_samp, pi_feat = (
            wxy_samp,
            wxy_feat,
        )  # shape nx_samp x ny_samp and nx_feat x ny_feat
        duals_samp = (
            nx.zeros(nx_samp, type_as=X),
            nx.zeros(ny_samp, type_as=Y),
        )  # shape nx_samp, ny_samp
        duals_feat = (
            nx.zeros(nx_feat, type_as=X),
            nx.zeros(ny_feat, type_as=Y),
        )  # shape nx_feat, ny_feat
    else:
        pi_samp, pi_feat = warmstart["pi_sample"], warmstart["pi_feature"]
        duals_samp, duals_feat = warmstart["duals_sample"], warmstart["duals_feature"]

    if prior is not None:
        # prior sample is the plan obtained from OT
        # prior feature is the initial adjcaent matrix
        pi_samp, pi_feat = prior["prior_samp"], prior["prior_feat"]

    # pi_feat = prior_feat
    # initialize log
    list_coot = [float("inf")]
    err = tol_bcd + 1e-3

    for idx in range(nits_bcd):
        pi_samp_prev = nx.copy(pi_samp)
        # update sample coupling
        ot_cost = cdist(X, Y @ pi_feat.T, metric="correlation")
        if eps_samp > 0 and prior is not None:
            pi_samp, dict_log = sinkhorn_knopp_unbalanced(
                a=wx_samp,
                b=wy_samp,
                M=ot_cost,
                reg=eps_samp,
                reg_m = reg_samp, 
                prior=pi_samp,
                # methods="samp",
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True, 
                warmstart=duals_samp,
            )
            duals_samp = (dict_log["logu"], dict_log["logv"])
            coot_samp = nx.sum(ot_cost * pi_samp)
            # print(evaluate(pi_samp))
        elif eps_samp > 0 and prior is None:
            pi_samp, dict_log = sinkhorn(
                a=wx_samp,
                b=wy_samp,
                M=ot_cost,
                reg=eps_samp,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_samp,
            )
            duals_samp = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))

        else:
            warnings.warn("Epsilon is needed for a continuous sanmple-level transport plan. ")
            return

        # update feature coupling
        A_squeezed = nx.squeeze(X.T)
        B_squeezed = nx.squeeze(Y.T @ pi_samp.T)
        jitter = 1e-10
        A_final = A_squeezed + np.random.randn(*A_squeezed.shape) * jitter
        B_final = B_squeezed + np.random.randn(*B_squeezed.shape) * jitter
        ot_cost = cdist(A_final, B_final, metric="correlation")
        if eps_feat > 0 and prior is not None:
            pi_feat, dict_log = sinkhorn_knopp_unbalanced(
                a=wx_feat,
                b=wy_feat,
                M=ot_cost,
                reg=eps_feat,
                reg_m = reg_feat,
                prior=pi_feat,
                # methods="feat", 
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_feat,
            )
            duals_feat = (dict_log["logu"], dict_log["logv"])
            # print("the sum of the feature OT plan at first row is: " + str(pi_feat[0].sum()))
            # print("the local information preservation is: " + str(pi_feat[0][0:4].sum() / (pi_feat[0].sum())))
            coot_feat = nx.sum(ot_cost * pi_feat)
        elif eps_feat > 0 and prior is None:
            pi_feat, dict_log = sinkhorn(
                a=wx_feat,
                b=wy_feat,
                M=ot_cost,
                reg=eps_feat,
                numItermax=nits_ot,
                stopThr=tol_sinkhorn,
                log=True,
                warmstart=duals_feat,
            )
            duals_feat = (nx.log(dict_log["u"]), nx.log(dict_log["v"]))
        else:
            warnings.warn("Epsilon is needed for a continuous feature-level transport plan. ")
            return

        if idx % eval_bcd == 0:
            # update error
            err = nx.sum(nx.abs(pi_samp - pi_samp_prev))

            # COOT part
            # Why minizing the overall cost on feature?
            # Try with sample,
            # coot = nx.sum(ot_cost * pi_feat)
            if alpha_samp != 0:
                coot = coot + alpha_samp * nx.sum(M_samp * pi_samp)
            # Entropic part
            # Also needs modify, although do not influence the general results, just for loss printing purpose.
            # if eps_samp != 0:
            #     coot = coot + eps_samp * nx.kl_div(pi_samp, prior_samp)
            # if eps_feat != 0:
            #     coot = coot + eps_feat * nx.kl_div(pi_feat, prior_feat)
            list_coot.append(coot_samp)

            if err < tol_bcd or abs(list_coot[-2] - list_coot[-1]) < early_stopping_tol:
                break

            if verbose:
                print(
                    "CO-Optimal Transport cost at iteration {}: sample-level OT cost:{}, feature-level OT cost:{}".format(
                        idx + 1, coot_samp, coot_feat
                    )
                )

    # sanity check
    if nx.sum(nx.isnan(pi_samp)) > 0 or nx.sum(nx.isnan(pi_feat)) > 0:
        warnings.warn("There is NaN in coupling.")

    if log:
        dict_log = {
            "duals_sample": duals_samp,
            "duals_feature": duals_feat,
            "distances": list_coot[1:],
        }

        return pi_samp, pi_feat, dict_log

    else:
        return pi_samp, pi_feat




def sinkhorn_knopp_unbalanced(
    a,
    b,
    M,
    reg,
    reg_m,
    prior,
    reg_type="kl",
    c=None,
    warmstart=None,
    numItermax=1000,
    stopThr=1e-6,
    verbose=False,
    log=False,
    **kwargs,
):
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    reg_m1, reg_m2 = reg_m, reg_m

    if log:
        dict_log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        if n_hists:
            u = nx.ones((dim_a, 1), type_as=M)
            v = nx.ones((dim_b, n_hists), type_as=M)
            a = a.reshape(dim_a, 1)
        else:
            u = nx.ones(dim_a, type_as=M)
            v = nx.ones(dim_b, type_as=M)
    else:
        u, v = nx.exp(warmstart[0]), nx.exp(warmstart[1])

    if reg_type.lower() == "entropy":
        warnings.warn(
            "If reg_type = entropy, then the matrix c is overwritten by the one matrix."
        )
        c = nx.ones((dim_a, dim_b), type_as=M)

    if n_hists:
        K = nx.exp(-M / reg)
    else:
        c = prior.copy()
        c = nx.maximum(c, 1e-10)
        K = nx.exp(-M / reg) * c

    fi_1 = reg_m1 / (reg_m1 + reg) if reg_m1 != float("inf") else 1
    fi_2 = reg_m2 / (reg_m2 + reg) if reg_m2 != float("inf") else 1

    err = 1.0

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fi_1
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fi_2
        

        if (
            nx.any(Ktu == 0.0)
            or nx.any(nx.isnan(u))
            or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u))
            or nx.any(nx.isinf(v))
        ):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn("Numerical errors at iteration %s" % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.0
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.0
        )
        err = 0.5 * (err_u + err_v)
        if log:
            dict_log["err"].append(err)
            if verbose:
                if i % 50 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(i, err))
        if err < stopThr:
            break

    # print(f"Sinkhorn Ends at {i} iterations.")
    if log:
        dict_log["logu"] = nx.log(u + 1e-300)
        dict_log["logv"] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum("ik,ij,jk,ij->k", u, K, v, M)
        if log:
            return res, dict_log
        else:
            return res

    else:  # return OT matrix
        plan = u[:, None] * K * v[None, :]

        if log:
            linear_cost = nx.sum(plan * M)
            dict_log["cost"] = linear_cost

            total_cost = linear_cost + reg * nx.kl_div(plan, c)
            if reg_m1 != float("inf"):
                total_cost = total_cost + reg_m1 * nx.kl_div(nx.sum(plan, 1), a)
            if reg_m2 != float("inf"):
                total_cost = total_cost + reg_m2 * nx.kl_div(nx.sum(plan, 0), b)
            dict_log["total_cost"] = total_cost

            return plan, dict_log
        else:
            return plan




