"""
Utilities for downloading preset voices.
"""
import os
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional

# Verified audio + transcript pairs extracted from demo pages
PRESET_VOICES = [
    # === Zero-shot (Multi-language) ===
    {
        "name": "zh_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/zh/prompt_audio_4.wav",
        "transcript": "转任福建路转运判官。",
        "language": "zh", "gender": "female", "style": "neutral"
    },
    {
        "name": "zh_expressive_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/hard-zh/prompt_audio_4.wav",
        "transcript": "在中国鸦片泛滥的年代，不同材质的烟枪甚至成为了身份和地位的象征。",
        "language": "zh", "gender": "female", "style": "expressive"
    },
    {
        "name": "en_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/en/prompt_audio_2.wav",
        "transcript": "There is no lock but a golden key will open it.",
        "language": "en", "gender": "female", "style": "neutral"
    },
    {
        "name": "en_male_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/hard-en/prompt_audio_4.wav",
        "transcript": "And there were dunes, rocks, and plants that insisted on living where survival seemed impossible.",
        "language": "en", "gender": "male", "style": "expressive"
    },
    {
        "name": "ja_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/ja/prompt_audio_25.wav",
        "transcript": "来週、美容院で髪を切ろうと思っています。",
        "language": "jp", "gender": "female", "style": "neutral"
    },
    {
        "name": "ko_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/ko/prompt_audio_5.wav",
        "transcript": "그들이 집까지 왔을 때는 어슬어슬한 황혼이었다.",
        "language": "ko", "gender": "female", "style": "neutral"
    },
    {
        "name": "de_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/de/prompt_audio_1.wav",
        "transcript": "Zieht euch bitte draußen die Schuhe aus.",
        "language": "de", "gender": "female", "style": "neutral"
    },
    {
        "name": "es_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/es/prompt_audio_1.wav",
        "transcript": "Durante unos años, enseñó Física e Historia en el colegio de nobles de Parma.",
        "language": "es", "gender": "female", "style": "neutral"
    },
    {
        "name": "fr_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/fr/prompt_audio_1.wav",
        "transcript": "Ce dernier a évolué tout au long de l'histoire romaine.",
        "language": "fr", "gender": "female", "style": "neutral"
    },
    {
        "name": "it_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/it/prompt_audio_2.wav",
        "transcript": "Fin dall'inizio la sede episcopale è stata immediatamente soggetta alla Santa Sede.",
        "language": "it", "gender": "female", "style": "neutral"
    },
    {
        "name": "ru_female_1",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/zero-shot/ru/prompt_audio_3.wav",
        "transcript": "Неожиданно катастрофа приобрела глобальные масштабы.",
        "language": "ru", "gender": "female", "style": "neutral"
    },
    
    # === Emotional Voices ===
    {
        "name": "emotion_happy_en",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/happy/prompt_audio_25.wav",
        "transcript": "Great, yeah. I mean, it has been great, too. You know, some of these people must have seen me play before because they were requesting a bunch of my songs.",
        "language": "en", "gender": "female", "style": "happy"
    },
    {
        "name": "emotion_happy_zh",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/happy/prompt_audio_1.wav",
        "transcript": "终于去看运动会啦,舒畅啊!",
        "language": "zh", "gender": "female", "style": "happy"
    },
    {
        "name": "emotion_sad_en",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/sad/prompt_audio_1.wav",
        "transcript": "Born once every 100 years, dies in flames.",
        "language": "en", "gender": "female", "style": "sad"
    },
    {
        "name": "emotion_sad_zh",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/sad/prompt_audio_7.wav",
        "transcript": "红了鼻头的小丑,眼泪止不住的流,流到嘴边咽下悲伤。",
        "language": "zh", "gender": "female", "style": "sad"
    },
    {
        "name": "emotion_fearful_en",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/fearful/prompt_audio_1.wav",
        "transcript": "I... I'm really nervous about getting my hair cut here... What if it doesn't turn out the way I want? I... I don't know if I can go through with it.",
        "language": "en", "gender": "female", "style": "fearful"
    },
    {
        "name": "emotion_fearful_zh",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/fearful/prompt_audio_8.wav",
        "transcript": "不断进步的科技，是不是会让医生不再需要人类来担任呢？",
        "language": "zh", "gender": "female", "style": "fearful"
    },
    {
        "name": "emotion_angry_en",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/angry/prompt_audio_13.wav",
        "transcript": "The boy, O'brien, was specially maltreated.",
        "language": "en", "gender": "male", "style": "angry"
    },
    {
        "name": "emotion_angry_zh",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/angry/prompt_audio_2.wav",
        "transcript": "受到处罚你可不能怨别人,知道吗,臭小子!",
        "language": "zh", "gender": "male", "style": "angry"
    },
    {
        "name": "emotion_surprised_en",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/surprised/prompt_audio_1.wav",
        "transcript": "I can't believe it— the lions just broke out of their enclosure and are walking around freely!",
        "language": "en", "gender": "female", "style": "surprised"
    },
    {
        "name": "emotion_surprised_zh",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/emotion/surprised/prompt_audio_2.wav",
        "transcript": "真的吗？！每个人居然真的都有权利追求自己的幸福？！这真是太不可思议了！",
        "language": "zh", "gender": "female", "style": "surprised"
    },
    
    # === Chinese Dialects ===
    {
        "name": "dialect_cantonese",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/dialect/tbJ6z3v8qCQ_533_1600_24800_addLeadSil80_addTrailSil160_trim_db27.wav",
        "transcript": "但系，好明显唔系啦。",
        "language": "zh", "gender": "female", "style": "cantonese"
    },
    {
        "name": "dialect_dongbei",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/dialect/dongbei_wav000_0018_dongbei_dialect_4_237_800_61504_addLeadSil80_addTrailSil160_trim_db27.wav",
        "transcript": "我媳妇说：啥？玩愣？你说啥？我没听清，你再说一遍。",
        "language": "zh", "gender": "female", "style": "dongbei"
    },
    {
        "name": "dialect_tianjin",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/dialect/tianjin_wav000_0192_tianjin_dialect_3_64_0_53600_addLeadSil80_addTrailSil160_trim_db27.wav",
        "transcript": "就问问，这锣是哪儿的人告诉是天津的。",
        "language": "zh", "gender": "female", "style": "tianjin"
    },
    {
        "name": "dialect_sichuan",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/dialect/sichuan_wav000_0006_Speaker0001_Android_s1_025_9600_77600_addLeadSil80_addTrailSil160_trim_db27.wav",
        "transcript": "此次新增的两列车，是整个增车项目的首批。",
        "language": "zh", "gender": "female", "style": "sichuan"
    },
    {
        "name": "dialect_shanghai",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/dialect/wav038_0035_T0065G0615S0381_2400_42400_addLeadSil80_addTrailSil160_trim_db27.wav",
        "transcript": "没钞票侬凭啥爱我？",
        "language": "zh", "gender": "female", "style": "shanghai"
    },
    
    # === Cross-lingual (Speakers who can speak multiple languages) ===
    {
        "name": "crosslingual_zh_m",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/cross-lingual/zh_m.wav",
        "transcript": "至今为止，元气火箭总共发行了两张专辑。",
        "language": "zh", "gender": "male", "style": "neutral"
    },
    {
        "name": "crosslingual_en_m",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/cross-lingual/en_m.wav",
        "transcript": "Hey look, a flying pig!",
        "language": "en", "gender": "male", "style": "neutral"
    },
    {
        "name": "crosslingual_zh_f",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/cross-lingual/zh_f.wav",
        "transcript": "我说你这只大鸟，真是不讲理，我对你做什么了呀，你就要吞了我！",
        "language": "zh", "gender": "female", "style": "expressive"
    },
    {
        "name": "crosslingual_en_f",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/cross-lingual/en_f.wav",
        "transcript": "I am the ghost of Christmas present. You have never seen anything like me before.",
        "language": "en", "gender": "female", "style": "neutral"
    },
    
    # === Instructed Voices ===
    {
        "name": "instruct_neutral",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/c3_large/insturct/1/Neutral_ZH_prompt.wav",
        "transcript": "中立 出来野餐不要再用一次性木筷，因为这是浪费木材。",
        "language": "zh", "gender": "female", "style": "neutral"
    },
    {
        "name": "instruct_angry",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/c3_large/insturct/2/Angry_ZH_prompt.wav",
        "transcript": "生气 刚才还好好的，一眨眼又消失了，真的是要气死我了。",
        "language": "zh", "gender": "female", "style": "angry"
    },
    {
        "name": "instruct_happy",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/c3_large/insturct/3/Happy_ZH_prompt.wav",
        "transcript": "高兴 能和大家在一起，我好开心啊。",
        "language": "zh", "gender": "female", "style": "happy"
    },
    
    # === Mixed-lingual Speaker ===
    {
        "name": "mixedlingual_speaker",
        "url": "https://funaudiollm.github.io/cosyvoice3/audio/prompt/mix/clone_jr.WAV",
        "transcript": "今天我们看到模型的本质，其实在很多时候是今天把我们人类的知识能够有效的聚集起来。能够成为今天我们一个重要的一个智慧体...",
        "language": "zh", "gender": "male", "style": "speaker"
    },
    
    # === VoxCPM Official Voices ===
    {
        "name": "ben",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/zeroshot/prompt/Ben_promptvn.wav",
        "transcript": "So it may be that you would prefer to forego my secret rather than consent to becoming a prisoner here for what might be several days.",
        "language": "en", "gender": "male", "style": "character"
    },
    {
        "name": "trump",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/zeroshot/prompt/trump_promptvn.wav",
        "transcript": "In short, we embarked on a mission to make America great again for all Americans.",
        "language": "en", "gender": "male", "style": "celebrity"
    },
    {
        "name": "andy_lau",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/zeroshot/prompt/dehua_promptvn.wav",
        "transcript": "所以我觉得这些成功的电影他都很真诚，而且很有生命力。他就跟当年的那个0号的那个一模一样。",
        "language": "zh", "gender": "male", "style": "celebrity"
    },
    {
        "name": "jia_ling",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/zeroshot/prompt/jialing_promptvn.wav",
        "transcript": "跟观众分享我人生的感悟。因为我们都是只活一次，我们也都是第一次活，我们也不知道该怎么活着。",
        "language": "zh", "gender": "female", "style": "celebrity"
    },
    {
        "name": "wu_jing",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/math/prompt/prompt_wujing.wav",
        "transcript": "坦克你没有后视镜的，枪炮是不长眼的，还有黑哥们儿的语言是不通的。",
        "language": "zh", "gender": "male", "style": "celebrity"
    },
    {
        "name": "meiyangyang",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/math/prompt/prompt_meiyangyang.wav",
        "transcript": "沸羊羊，你吃东西能不能斯文一点啊？",
        "language": "zh", "gender": "female", "style": "character"
    },
    {
        "name": "cai_xukun",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/phoneme/prompt/prompt_cai.wav",
        "transcript": "你干嘛哎哟。",
        "language": "zh", "gender": "male", "style": "celebrity"
    },
    {
        "name": "baoerjie",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/dialect_zeroshot/prompt_wav/baoerjie.wav",
        "transcript": "他们总说我瓜，其实我一点儿都不瓜，大多时候我都机智的一笔。",
        "language": "zh", "gender": "female", "style": "dialect"
    },
    {
        "name": "dialect_guangxi",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/dialect_zeroshot/prompt_wav/guangxi1.wav",
        "transcript": "算命先生说我24岁会黄袍加身，餐餐都有大鱼大肉为伴。我信你个鬼，你这个糟老头子坏的很。",
        "language": "zh", "gender": "male", "style": "dialect"
    },
    {
        "name": "dialect_cantonese_vox",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/dialect_zeroshot/prompt_wav/yueyu1.wav",
        "transcript": "着西装打呔，攞大哥电话有咩用啊？啊？跟着这些大佬，吔屎啊你。",
        "language": "zh", "gender": "male", "style": "dialect"
    },
    {
        "name": "dialect_henan",
        "url": "https://openbmb.github.io/VoxCPM-demopage/audio/dialect_zeroshot/prompt_wav/henanhua.wav",
        "transcript": "我感觉说河南话不影响我的颜值啊，我自己听不出来，恁感觉呢，恁感觉说河南话影响我的颜值吗？恁感觉呢姐妹们。",
        "language": "zh", "gender": "female", "style": "dialect"
    }
]

def download_presets(output_dir: str = "presets/voices", verbose: bool = True):
    """Download all preset voices with verified transcripts.
    
    Args:
        output_dir: Directory to save voices (default: "presets/voices")
        verbose: Whether to print progress (default: True)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {}
    
    if verbose:
        print(f"Downloading {len(PRESET_VOICES)} preset voices with verified transcripts...")
        print(f"Output directory: {output_path.absolute()}")
        print()
    
    for i, voice in enumerate(PRESET_VOICES, 1):
        name = voice["name"]
        url = voice["url"]
        filename = f"{name}.wav"
        filepath = output_path / filename
        
        if verbose:
            print(f"[{i}/{len(PRESET_VOICES)}] {name}...", end=" ")
        
        try:
            if filepath.exists():
                if verbose: print("⏭️ exists")
            else:
                urllib.request.urlretrieve(url, filepath)
                if verbose: print("✅ downloaded")
            
            metadata[name] = {
                "audio": filename,
                "language": voice["language"],
                "gender": voice["gender"],
                "style": voice["style"],
                "transcript": voice["transcript"],
            }
            
        except Exception as e:
            if verbose: print(f"❌ failed: {e}")
    
    # Save metadata
    metadata_path = output_path.parent / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print()
        print(f"✅ Downloaded {len(metadata)} voices")
        print(f"📝 Metadata saved to: {metadata_path}")
