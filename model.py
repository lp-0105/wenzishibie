import paddle
import paddle.nn as nn

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
