import paddle
import paddle.nn as nn
import numpy as np

class TPS_SpatialTransformer(nn.Layer):
    def __init__(self, F=20, I_size=(32, 100), I_r_size=(32, 100), i_channel_num=1):
        """ 
        TPS 空间变换器:
        F: 控制点数量 (默认 20, 即上下各 10 个)
        I_size: 输入图像尺寸
        I_r_size: 输出(矫正后)图像尺寸
        """
        super(TPS_SpatialTransformer, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.i_channel_num = i_channel_num
        
        # 1. 定位网络 (Localization Network): 预测控制点偏移
        self.LocNet = nn.Sequential(
            nn.Conv2D(i_channel_num, 64, 3, 1, 1), nn.BatchNorm2D(64), nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(64, 128, 3, 1, 1), nn.BatchNorm2D(128), nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(128, 256, 3, 1, 1), nn.BatchNorm2D(256), nn.ReLU(),
            nn.MaxPool2D(2, 2),
            nn.Conv2D(256, 512, 3, 1, 1), nn.BatchNorm2D(512), nn.ReLU(),
            nn.AdaptiveAvgPool2D(1)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, F * 2) # 输出 F 个点的 (x, y) 坐标
        )
        
        # 初始化为恒等变换 (即不偏移)
        self.fc_loc[-1].weight.set_value(paddle.zeros([256, F * 2]))
        # 默认基准控制点 (通常沿边界分布)
        ctrl_pts_x = np.linspace(-1.0, 1.0, F // 2)
        ctrl_pts_y_top = np.ones(F // 2) * -1.0
        ctrl_pts_y_bottom = np.ones(F // 2) * 1.0
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).flatten()
        self.fc_loc[-1].bias.set_value(paddle.to_tensor(initial_bias, dtype='float32'))

        # 预计算基准网格 (常态坐标)
        self.register_buffer('grid', self._calculate_grid(I_r_size))
        # 预计算用于解 TPS 方程的矩阵
        self.register_buffer('K_inv', self._calculate_K_inv(F, initial_bias.reshape([F, 2])))

    def _calculate_grid(self, I_r_size):
        h, w = I_r_size
        x = paddle.linspace(-1.0, 1.0, w)
        y = paddle.linspace(-1.0, 1.0, h)
        grid_y, grid_x = paddle.meshgrid(y, x)
        return paddle.stack([grid_x, grid_y], axis=2).reshape([-1, 2]) # [N, 2]

    def _calculate_K_inv(self, F, ctrl_pts):
        # TPS 核矩阵计算: 使用 numpy 在 CPU 上完成，避免 GPU 上的 dtype/维度 冲突
        ctrl_pts = np.array(ctrl_pts, dtype='float32')
        def R2_log_R_np(r2):
            return r2 * np.log(r2 + 1e-6)
        
        K = np.zeros([F, F], dtype='float32')
        for i in range(F):
            for j in range(F):
                r2 = np.sum((ctrl_pts[i] - ctrl_pts[j])**2)
                K[i, j] = R2_log_R_np(r2)
        
        P = np.concatenate([np.ones([F, 1], dtype='float32'), ctrl_pts], axis=1)
        L = np.concatenate([
            np.concatenate([K, P], axis=1),
            np.concatenate([P.transpose(), np.zeros([3, 3], dtype='float32')], axis=1)
        ], axis=0)
        return paddle.inverse(paddle.to_tensor(L, dtype='float32'))

    def forward(self, x):
        batch_size = x.shape[0]
        # 1. 预测控制点 C'
        points = self.fc_loc(self.LocNet(x).reshape([batch_size, -1])) # [B, F*2]
        points = points.reshape([batch_size, self.F, 2])
        
        # 2. 计算变换矩阵 T (求解 TPS 方程: L * T = [C'; 0])
        zero_pad = paddle.zeros([batch_size, 3, 2])
        Y = paddle.concat([points, zero_pad], axis=1)
        T = paddle.matmul(self.K_inv, Y) # [B, F+3, 2]
        
        # 3. 映射到输出网格坐标
        # 计算网格点与基准控制点的距离核
        grid_flat = self.grid # [N, 2]
        n_grid = grid_flat.shape[0]
        
        # 预计算基准点 (这里简化处理，实际中应使用 register_buffer 存好的 ctrl_pts)
        ctrl_pts = self.fc_loc[-1].bias.reshape([self.F, 2])
        
        # 距离核 |p - p_i|^2 * log|p - p_i|
        # 稍微重塑进行广播计算
        grid_ext = grid_flat.unsqueeze(1) # [N, 1, 2]
        ctrl_ext = ctrl_pts.unsqueeze(0)   # [1, F, 2]
        diff = grid_ext - ctrl_ext         # [N, F, 2]
        r2 = paddle.sum(diff**2, axis=2)   # [N, F]
        K_grid = r2 * paddle.log(r2 + 1e-6)
        
        # 拼接 P 矩阵 [1, x, y]
        P_grid = paddle.concat([paddle.ones([n_grid, 1]), grid_flat], axis=1) # [N, 3]
        L_grid = paddle.concat([K_grid, P_grid], axis=1) # [N, F+3]
        
        # 计算采样网格
        target_grid = paddle.matmul(L_grid, T) # [B, N, 2]
        target_grid = target_grid.reshape([batch_size, self.I_r_size[0], self.I_r_size[1], 2])
        
        # 4. 采样变换
        out = paddle.nn.functional.grid_sample(x, target_grid, padding_mode='zeros', align_corners=True)
        return out

class SEResidualBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2D(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(out_channels // reduction, out_channels, bias_attr=False),
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1, stride),
                nn.BatchNorm2D(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        b, c, _, _ = out.shape
        y = self.avg_pool(out).reshape([b, c])
        y = self.fc(y).reshape([b, c, 1, 1])
        out = out * y 
        out += identity
        return self.relu(out)

class TransformerOCR(nn.Layer):
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=4):
        super(TransformerOCR, self).__init__()
        # 0. TPS Pre-processing Layer
        # 输入通常是训练脚本里 resize 后的尺度 (H=32, W=320)
        self.tps = TPS_SpatialTransformer(F=20, I_size=(32, 320), I_r_size=(32, 320), i_channel_num=1)

        self.conv1 = nn.Sequential(
            nn.Conv2D(1, 64, 3, stride=2, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.layer1 = SEResidualBlock(64, 128, stride=2)
        self.layer2 = SEResidualBlock(128, 128)
        self.layer3 = SEResidualBlock(128, 256, stride=(2, 1))
        self.layer4 = SEResidualBlock(256, 256)
        
        self.H, self.W = 4, 40
        self.final_pool = nn.AdaptiveAvgPool2D((self.H, self.W)) 
        self.final_conv = nn.Sequential(
            nn.Conv2D(256, d_model, kernel_size=1), 
            nn.BatchNorm2D(d_model),
            nn.ReLU()
        )
        self.pos_embedding = self.create_parameter(
            shape=[1, self.H * self.W, d_model], 
            default_initializer=nn.initializer.TruncatedNormal(std=0.02)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Branch A: CTC Head
        self.fc = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(d_model, num_classes)
        )
        
        # Branch B: Attention Head (Semantic Enhancer)
        self.embedding = nn.Embedding(num_classes, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_model*4, dropout=0.2)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.att_fc = nn.Linear(d_model, num_classes)

    def forward(self, x, targets=None):
        # 1. Image Rectification
        x = self.tps(x)

        # 2. SE-ResNet Feature Extraction
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_pool(x) 
        x = self.final_conv(x)
        b, c, h, w = x.shape
        memory = x.reshape([b, c, h * w]).transpose([0, 2, 1]) 
        memory = memory + self.pos_embedding
        memory = self.transformer(memory) 
        
        # CTC Branch
        ctc_feat = memory.reshape([b, h, w, c]).mean(axis=1)
        ctc_out = self.fc(ctc_feat)
        
        # Attention Branch (Only for Training)
        att_out = None
        if targets is not None:
            tgt_emb = self.embedding(targets)
            att_out = self.decoder(tgt_emb, memory)
            att_out = self.att_fc(att_out)
            
        return ctc_out, att_out
