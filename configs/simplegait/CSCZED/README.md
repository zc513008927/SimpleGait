# SimpleGait fusion variants

The LPF-family variants share exactly the same fusion pipeline. The only
difference is whether a channel-attention block is inserted **after
concatenation and before the final 1x1 projection**.

```text
Modality 1 -> 1x1 projection --\
                              concat -> optional attention -> 1x1 projection -> fused feature
Modality 2 -> 1x1 projection --/
```

Available settings:

```yaml
fusion_type: lpf
```

Pure LPF. No extra operation is applied to `concat_features`.

```yaml
fusion_type: lpf_se
```

Uses `SEBlock` on `concat_features`.

```yaml
fusion_type: lpf_eca
```

Uses `ECABlock` on `concat_features`.

The key implementation is:

```python
concat_features = torch.cat([modality1_feat, modality2_feat], dim=1)
concat_features = self.channel_attention(concat_features)
fused_feat = self.fusion_conv(concat_features)
```

For pure LPF, `self.channel_attention` is `nn.Identity()`, so this line changes
nothing. For LPF+SE or LPF+ECA, the corresponding channel-attention block is
applied at exactly the same position.
