import torch
import torch.nn as nn
from model.model_eeg import Unimodal  # 导入EEG特征提取模型
from model.model_face import EnhancedLandmarkToEEG  # 导入面部特征提取模型
from model.EEGNet import EEG_net
from model.FaceNet import Face_net
from model.teacher_model import eeg_Model
from model.teacher_model import face_Model
from model.transformer import Inter_modal_Attention_Interaction
from aug_nor import augment_eeg_weak, augment_eeg_strong, augment_face_weak, augment_face_strong


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()

        self.opt = opt
        self.eeg_extractor = Unimodal(opt)  # EEG特征提取器
        self.face_extractor = EnhancedLandmarkToEEG()  # 面部特征提取器
        self.eeg_encoder = EEG_net(opt)
        self.face_encoder = Face_net(opt)
        self.cross_attention_layer = Inter_modal_Attention_Interaction(opt)

        # 加载 EEG 教师模型（冻结参数）
        self.eeg_teacher = eeg_Model(opt)
        self.face_teacher = face_Model(opt)
        eeg_teacher_path = '/home/user/eeg_teacher_model_fold5_best_acc_0.9205.pth'
        face_teacher_path = '/home/user/face_teacher_model_fold4_best_acc_0.8892.pth'

        eeg_checkpoint = torch.load(eeg_teacher_path, map_location='cuda')
        face_checkpoint = torch.load(face_teacher_path, map_location='cuda')

        # 前缀要移除的列表，可以扩展更多
        prefixes_to_remove = ['model.', 'trainer.']
        # 处理EEG模型的状态字典
        eeg_state_dict = eeg_checkpoint['model_state_dict']
        new_eeg_state_dict = {}
        for k, v in eeg_state_dict.items():
            new_key = k
            for prefix in prefixes_to_remove:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]  # 移除前缀
            # break  # 一旦移除，跳出前缀检查
        new_eeg_state_dict[new_key] = v

        state_dict = face_checkpoint['model_state_dict']
        new_state_dict = {}
        prefix = "trainer."  
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]  
            else:
                new_key = k
        new_state_dict[new_key] = v
        # 现在用新的state_dict加载模型
        self.face_teacher.load_state_dict(new_state_dict, strict=False)

        self.eeg_teacher.eval()
        self.face_teacher.eval()
        for p in self.eeg_teacher.parameters():
            p.requires_grad = False
        for p in self.face_teacher.parameters():
            p.requires_grad = False

    def forward(self, x1, x2):
        eeg_feat = self.eeg_extractor(x1)  # 使用model_eeg提取特征
        face_feat = self.face_extractor(x2)  # 使用model_face提取特征
        # 数据增强
        eeg_weak = augment_eeg_weak(eeg_feat)
        eeg_strong = augment_eeg_strong(eeg_feat)
        face_weak = augment_face_weak(face_feat)
        face_strong = augment_face_strong(face_feat)
      

        eeg_emd_s = self.eeg_encoder(eeg_strong)
        eeg_emd_w = self.eeg_encoder(eeg_weak)
       

        face_emd_s = self.face_encoder(face_strong)
        face_emd_w = self.face_encoder(face_weak)
        '''
            eeg_emd_s.shape : [176, 4, 32, 60]
            eeg_emd_w.shape : [176, 4, 32, 60]
            ..
        '''
      

        with torch.no_grad():
            eeg_teacher = self.eeg_teacher.eeg_encoder(eeg_feat).view(eeg_feat.size(0), -1)
            face_teacher = self.face_teacher.face_encoder(face_feat).view(face_feat.size(0), -1)

        eeg_s_flat = eeg_emd_s.view(eeg_emd_s.size(0), -1)
        face_s_flat = face_emd_s.view(face_emd_s.size(0), -1)
        eeg_w_flat = eeg_emd_w.view(eeg_emd_w.size(0), -1)
        face_w_flat = face_emd_w.view(face_emd_w.size(0), -1)

    

        loss_cc_s, loss_uni_s, eeg_ratio, face_ratio = counterfactual_distill(
            eeg_s_flat, face_s_flat, eeg_teacher, face_teacher
        )
        loss_cc_w, loss_uni_w, eeg_ratio, face_ratio = counterfactual_distill(
            eeg_w_flat, face_w_flat, eeg_teacher, face_teacher
        )

      
        fused_eeg = torch.cat([eeg_emd_s, eeg_emd_w], dim=-1)
        fused_face = torch.cat([face_emd_s, face_emd_w], dim=-1)
       
        # #######融合
        # ##cat
        # # fused_emd = torch .cat([fused_eeg,fused_face], dim= -1)
        #
        # # 相加除以2
        # # fused_emd = (fused_eeg + fused_face)/2
        #
        # ###跨融合
        #
        fused_eeg = fused_eeg.view(fused_eeg.size(0), -1, fused_eeg.size(3))  # [B, 128, 60]
        fused_face = fused_face.view(fused_face.size(0), -1, fused_face.size(3))  # [B, 128, 60]
        
        fused_eeg = fused_eeg.permute(0, 2, 1)
        fused_face = fused_face.permute(0, 2, 1)
        
        fuse1 = self.cross_attention_layer(fused_eeg, fused_face)
        fuse2 = self.cross_attention_layer(fused_face, fused_eeg)
        fused_emd = torch.cat([fuse1, fuse2], dim=-1)

        fused_emd = fused_emd.reshape(fused_emd.size(0), -1)


        return {
            "fused_emd": fused_emd,
            "loss_cc_s": loss_cc_s,
            "loss_cc_w": loss_cc_w,
            "loss_uni_s": loss_uni_s,
            "loss_uni_w": loss_uni_w,
            "z_eeg_s": eeg_emd_s,
            "z_eeg_w": eeg_emd_w,
            "z_face_s": face_emd_s,
            "z_face_w": face_emd_w,
        }


def KLDiverge(tpreds, spreds, distillTemp):
    tpreds = (tpreds / distillTemp).sigmoid()
    spreds = (spreds / distillTemp).sigmoid()
    return -(tpreds * (spreds + 1e-8).log() + (1 - tpreds) * (1 - spreds + 1e-8).log()).mean()


def get_ranking_score(u_embeds, i_embeds, pos, neg):
    pos_score = (u_embeds * i_embeds[pos]).sum(dim=-1)
    neg_score = (u_embeds * i_embeds[neg]).sum(dim=-1)
    return pos_score - neg_score


def counterfactual_distill(student_eeg, student_face, teacher_eeg, teacher_face):
    B = student_eeg.size(0)
    anchor_idx = torch.arange(B, device=student_eeg.device)
    pos_idx = anchor_idx
    neg_idx = torch.roll(anchor_idx, shifts=1)
    kl_param = 5
    temperature = 0.1

    # diff
    eeg_teacher_rank = get_ranking_score(teacher_eeg, teacher_eeg, pos_idx, neg_idx)
    face_teacher_rank = get_ranking_score(teacher_face, teacher_face, pos_idx, neg_idx)

    eeg_student_rank = get_ranking_score(student_eeg, student_eeg, pos_idx, neg_idx)
    face_student_rank = get_ranking_score(student_face, student_face, pos_idx, neg_idx)

    # kd loss
    eeg_kd_loss = torch.clamp(eeg_teacher_rank - eeg_student_rank, 0)
    face_kd_loss = torch.clamp(face_teacher_rank - face_student_rank, 0)

    eeg_kd_loss += kl_param * KLDiverge(eeg_teacher_rank, eeg_student_rank, temperature)
    face_kd_loss += kl_param * KLDiverge(face_teacher_rank, face_student_rank, temperature)

    # kd weight
    joint_score = (student_eeg * student_face).sum(dim=-1) - \
                  (student_eeg * torch.roll(student_face, shifts=1, dims=0)).sum(dim=-1)

    eeg_score = (student_eeg * teacher_face).sum(dim=-1) - \
                (student_eeg * torch.roll(teacher_face, shifts=1, dims=0)).sum(dim=-1)

    face_score = (student_face * teacher_eeg).sum(dim=-1) - \
                 (student_face * torch.roll(teacher_eeg, shifts=1, dims=0)).sum(dim=-1)

    coeff_eeg = torch.clamp((joint_score - face_score) / (eeg_score + 1e-8), 1e-8, 10)
    coeff_face = torch.clamp((joint_score - eeg_score) / (face_score + 1e-8), 1e-8, 10)
    denom = coeff_eeg + coeff_face
    eeg_ratio = 1 - (coeff_eeg - coeff_face) / denom
    face_ratio = 2 - eeg_ratio

    loss_cc = (eeg_ratio * eeg_kd_loss + face_ratio * face_kd_loss).mean()
    loss_uni = (eeg_kd_loss + face_kd_loss).mean()

    return loss_cc, loss_uni, eeg_ratio.mean().item(), face_ratio.mean().item()
