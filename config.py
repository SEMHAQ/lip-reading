# config.py
# 映射 IEMOCAP 和 MELD 的原始标签到你的理论模型
EMO_MAP = {
    'AW': { # 唤醒度：高、中、低
        'high': ['ang', 'exc', 'fru', 'hap'], # 生气、兴奋、挫败、开心
        'mid':  ['sur', 'dis'],               # 惊讶、厌恶
        'low':  ['sad', 'neu', 'fea']         # 难过、中性、恐惧
    },
    'PN': { # 发音动作干扰：A(嘴型)、B(唇齿)、C(韵律)、N(中性)
        'Type_A': ['hap', 'exc', 'sur'],      # 快乐/惊讶时嘴部张幅大
        'Type_B': ['ang', 'dis', 'fru'],      # 愤怒/厌恶时唇部紧绷或龇牙
        'Type_C': ['sad', 'fea'],             # 难过/恐惧时节奏变慢或颤抖
        'Type_N': ['neu']                     # 中性
    },
    'AC': { # 情感动机：趋近、回避、攻击、抑制、基准
        'Approach': ['hap', 'exc'],
        'Avoid':    ['fea', 'dis'],
        'Attack':   ['ang', 'fru'],
        'Inhibit':  ['sad'],
        'Base':     ['neu']
    }
}
