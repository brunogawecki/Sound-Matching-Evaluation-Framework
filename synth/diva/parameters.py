"""Diva's parameter names, module-qualified and in plugin index order.

Diva reports 2362 VST3 parameters, but 2080 of those are MIDI CC passthroughs: the real
parameter space is the 281 below, at indices 0..280. Their plugin-reported names are **not
unique** -- six parameters are called 'Rate', five 'Model', six 'Wet' -- so a bare name does not
address a Diva parameter and cannot key a ParameterSpace. Every name here is therefore qualified
with the module that owns it ('LFO1.Rate', 'VCF1.Model'), which is unique.

VST3 never reports the owning module, so this table cannot be rebuilt from the live plugin. It
was derived from ``code/synth/diva_params.txt`` in the Flow Synthesizer repository
(acids-ircam/flow_synthesizer, Esling et al.), whose list is from an earlier Diva whose indices
have since drifted, realigned onto this Diva by matching parameter names and assigning the
handful of parameters added since to the module block containing them.

The list is static on purpose. A Diva update that inserts a parameter shifts every index after
it, which would silently repoint the whole parameter space; ``tests/test_diva_parameters.py``
fails when the plugin stops agreeing with this table.
"""
from typing import Dict, List

# Index in this list == the plugin's parameter index.
DIVA_PARAMETER_NAMES: List[str] = [
    # main
    "main.Output",  # 0
    "main.Active #FX1",  # 1
    "main.Active #FX2",  # 2
    # PCore
    "PCore.LED Colour",  # 3
    # VCC
    "VCC.Voices",  # 4
    "VCC.Voice Stack",  # 5
    "VCC.Mode",  # 6
    "VCC.GlideMode",  # 7
    "VCC.Glide",  # 8
    "VCC.Glide2",  # 9
    "VCC.GlideRange",  # 10
    "VCC.PitchBend Up",  # 11
    "VCC.PitchBend Down",  # 12
    "VCC.TuningMode",  # 13
    "VCC.Transpose",  # 14
    "VCC.FineTuneCents",  # 15
    "VCC.Note Priority",  # 16
    "VCC.MultiCore",  # 17
    # OPT
    "OPT.Accuracy",  # 18
    "OPT.OfflineAcc",  # 19
    "OPT.TuneSlop",  # 20
    "OPT.CutoffSlop",  # 21
    "OPT.GlideSlop",  # 22
    "OPT.PWSlop",  # 23
    "OPT.EnvrateSlop",  # 24
    "OPT.V1Mod",  # 25
    "OPT.V2Mod",  # 26
    "OPT.V3Mod",  # 27
    "OPT.V4Mod",  # 28
    "OPT.V5Mod",  # 29
    "OPT.V6Mod",  # 30
    "OPT.V7Mod",  # 31
    "OPT.V8Mod",  # 32
    # ENV1
    "ENV1.Attack",  # 33
    "ENV1.Decay",  # 34
    "ENV1.Sustain",  # 35
    "ENV1.Release",  # 36
    "ENV1.Velocity",  # 37
    "ENV1.Model",  # 38
    "ENV1.Trigger",  # 39
    "ENV1.Quantise",  # 40
    "ENV1.Curve",  # 41
    "ENV1.Release On",  # 42
    "ENV1.KeyFollow",  # 43
    # ENV2
    "ENV2.Attack",  # 44
    "ENV2.Decay",  # 45
    "ENV2.Sustain",  # 46
    "ENV2.Release",  # 47
    "ENV2.Velocity",  # 48
    "ENV2.Model",  # 49
    "ENV2.Trigger",  # 50
    "ENV2.Quantise",  # 51
    "ENV2.Curve",  # 52
    "ENV2.Release On",  # 53
    "ENV2.KeyFollow",  # 54
    # LFO1
    "LFO1.Sync",  # 55
    "LFO1.Restart",  # 56
    "LFO1.Waveform",  # 57
    "LFO1.Phase",  # 58
    "LFO1.Delay",  # 59
    "LFO1.DepthMod Src1",  # 60
    "LFO1.DepthMod Dpt1",  # 61
    "LFO1.Rate",  # 62
    "LFO1.FreqMod Src1",  # 63
    "LFO1.FreqMod Dpt",  # 64
    "LFO1.Polarity",  # 65
    # LFO2
    "LFO2.Sync",  # 66
    "LFO2.Restart",  # 67
    "LFO2.Waveform",  # 68
    "LFO2.Phase",  # 69
    "LFO2.Delay",  # 70
    "LFO2.DepthMod Src1",  # 71
    "LFO2.DepthMod Dpt1",  # 72
    "LFO2.Rate",  # 73
    "LFO2.FreqMod Src1",  # 74
    "LFO2.FreqMod Dpt",  # 75
    "LFO2.Polarity",  # 76
    # MOD
    "MOD.Quantise",  # 77
    "MOD.Slew Rate",  # 78
    "MOD.RectifySource",  # 79
    "MOD.InvertSource",  # 80
    "MOD.QuantiseSource",  # 81
    "MOD.LagSource",  # 82
    "MOD.AddSource1",  # 83
    "MOD.AddSource2",  # 84
    "MOD.MulSource1",  # 85
    "MOD.MulSource2",  # 86
    # OSC
    "OSC.Model",  # 87
    "OSC.Tune1",  # 88
    "OSC.Tune2",  # 89
    "OSC.Tune3",  # 90
    "OSC.Vibrato",  # 91
    "OSC.PulseWidth",  # 92
    "OSC.Shape1",  # 93
    "OSC.Shape2",  # 94
    "OSC.Shape3",  # 95
    "OSC.FM",  # 96
    "OSC.Sync2",  # 97
    "OSC.OscMix",  # 98
    "OSC.Volume1",  # 99
    "OSC.Volume2",  # 100
    "OSC.Volume3",  # 101
    "OSC.PulseShape",  # 102
    "OSC.SawShape",  # 103
    "OSC.SuboscShape",  # 104
    "OSC.Tune1ModSrc",  # 105
    "OSC.Tune1ModDepth",  # 106
    "OSC.Tune2ModSrc",  # 107
    "OSC.Tune2ModDepth",  # 108
    "OSC.PWModSrc",  # 109
    "OSC.PWModDepth",  # 110
    "OSC.ShapeSrc",  # 111
    "OSC.ShapeDepth",  # 112
    "OSC.Triangle1On",  # 113
    "OSC.Sine2On",  # 114
    "OSC.Saw1On",  # 115
    "OSC.Pwm1On",  # 116
    "OSC.Triangle2On",  # 117
    "OSC.Saw2On",  # 118
    "OSC.Pulse2On",  # 119
    "OSC.PWM2On",  # 120
    "OSC.Noise1On",  # 121
    "OSC.ShapeModel",  # 122
    "OSC.Sync3",  # 123
    "OSC.NoiseVol",  # 124
    "OSC.NoiseColor",  # 125
    "OSC.TuneModOsc1",  # 126
    "OSC.TuneModOsc2",  # 127
    "OSC.TuneModOsc3",  # 128
    "OSC.ShapeModOsc1",  # 129
    "OSC.ShapeModOsc2",  # 130
    "OSC.ShapeModOsc3",  # 131
    "OSC.TuneModMode",  # 132
    "OSC.EcoWave1",  # 133
    "OSC.EcoWave2",  # 134
    "OSC.RingmodPulse",  # 135
    "OSC.Drift",  # 136
    "OSC.FmModSrc",  # 137
    "OSC.FmModDepth",  # 138
    "OSC.NoiseVolModSrc",  # 139
    "OSC.NoiseVolModDepth",  # 140
    "OSC.DigitalShape2",  # 141
    "OSC.DigitalShape3",  # 142
    "OSC.DigitalShape4",  # 143
    "OSC.DigitalType1",  # 144
    "OSC.DigitalType2",  # 145
    "OSC.DigitalAntiAlias",  # 146
    # HPF
    "HPF.Model",  # 147
    "HPF.Frequency",  # 148
    "HPF.Resonance",  # 149
    "HPF.Revision",  # 150
    "HPF.KeyFollow",  # 151
    "HPF.FreqModSrc",  # 152
    "HPF.FreqModDepth",  # 153
    "HPF.Post-HPF Freq",  # 154
    # VCF1
    "VCF1.Model",  # 155
    "VCF1.Frequency",  # 156
    "VCF1.Resonance",  # 157
    "VCF1.FreqModSrc",  # 158
    "VCF1.FreqModDepth",  # 159
    "VCF1.FreqMod2Src",  # 160
    "VCF1.FreqMod2Depth",  # 161
    "VCF1.KeyFollow",  # 162
    "VCF1.FilterFM",  # 163
    "VCF1.LadderMode",  # 164
    "VCF1.LadderColor",  # 165
    "VCF1.SlnKyRevision",  # 166
    "VCF1.SvfMode",  # 167
    "VCF1.Feedback",  # 168
    "VCF1.ResModSrc",  # 169
    "VCF1.ResModDepth",  # 170
    "VCF1.FmAmountModSrc",  # 171
    "VCF1.FmAmountModDepth",  # 172
    "VCF1.FeedbackModSrc",  # 173
    "VCF1.FeedbackModDepth",  # 174
    "VCF1.ShapeMix",  # 175
    "VCF1.ShapeModSrc",  # 176
    "VCF1.ShapeModDepth",  # 177
    "VCF1.UhbieBandpass",  # 178
    # VCA1
    "VCA1.Pan",  # 179
    "VCA1.Volume",  # 180
    "VCA1.VCA",  # 181
    "VCA1.Modulation",  # 182
    "VCA1.ModDepth",  # 183
    "VCA1.PanModulation",  # 184
    "VCA1.PanModDepth",  # 185
    "VCA1.Mode",  # 186
    "VCA1.Offset",  # 187
    # Scope1
    "Scope1.Frequency",  # 188
    "Scope1.Scale",  # 189
    # FX1
    "FX1.Module",  # 190
    # Chrs1
    "Chrs1.Type",  # 191
    "Chrs1.Rate",  # 192
    "Chrs1.Depth",  # 193
    "Chrs1.Wet",  # 194
    # Phase1
    "Phase1.Type",  # 195
    "Phase1.Rate",  # 196
    "Phase1.Feedback",  # 197
    "Phase1.Stereo",  # 198
    "Phase1.Sync",  # 199
    "Phase1.Phase",  # 200
    "Phase1.Wet",  # 201
    "Phase1.Depth",  # 202
    "Phase1.Center",  # 203
    # Plate1
    "Plate1.PreDelay",  # 204
    "Plate1.Diffusion",  # 205
    "Plate1.Damp",  # 206
    "Plate1.Decay",  # 207
    "Plate1.Size",  # 208
    "Plate1.Dry",  # 209
    "Plate1.Wet",  # 210
    # Delay1
    "Delay1.Left Delay",  # 211
    "Delay1.Center Delay",  # 212
    "Delay1.Right Delay",  # 213
    "Delay1.Side Vol",  # 214
    "Delay1.Center Vol",  # 215
    "Delay1.Feedback",  # 216
    "Delay1.HP",  # 217
    "Delay1.LP",  # 218
    "Delay1.Dry",  # 219
    "Delay1.Wow",  # 220
    # Rtary1
    "Rtary1.Mode",  # 221
    "Rtary1.Mix",  # 222
    "Rtary1.Balance",  # 223
    "Rtary1.Drive",  # 224
    "Rtary1.Stereo",  # 225
    "Rtary1.Out",  # 226
    "Rtary1.Slow",  # 227
    "Rtary1.Fast",  # 228
    "Rtary1.RiseTime",  # 229
    "Rtary1.Controller",  # 230
    # FX2
    "FX2.Module",  # 231
    # Chrs2
    "Chrs2.Type",  # 232
    "Chrs2.Rate",  # 233
    "Chrs2.Depth",  # 234
    "Chrs2.Wet",  # 235
    # Phase2
    "Phase2.Type",  # 236
    "Phase2.Rate",  # 237
    "Phase2.Feedback",  # 238
    "Phase2.Stereo",  # 239
    "Phase2.Sync",  # 240
    "Phase2.Phase",  # 241
    "Phase2.Wet",  # 242
    "Phase2.Depth",  # 243
    "Phase2.Center",  # 244
    # Plate2
    "Plate2.PreDelay",  # 245
    "Plate2.Diffusion",  # 246
    "Plate2.Damp",  # 247
    "Plate2.Decay",  # 248
    "Plate2.Size",  # 249
    "Plate2.Dry",  # 250
    "Plate2.Wet",  # 251
    # Delay2
    "Delay2.Left Delay",  # 252
    "Delay2.Center Delay",  # 253
    "Delay2.Right Delay",  # 254
    "Delay2.Side Vol",  # 255
    "Delay2.Center Vol",  # 256
    "Delay2.Feedback",  # 257
    "Delay2.HP",  # 258
    "Delay2.LP",  # 259
    "Delay2.Dry",  # 260
    "Delay2.Wow",  # 261
    # Rtary2
    "Rtary2.Mode",  # 262
    "Rtary2.Mix",  # 263
    "Rtary2.Balance",  # 264
    "Rtary2.Drive",  # 265
    "Rtary2.Stereo",  # 266
    "Rtary2.Out",  # 267
    "Rtary2.Slow",  # 268
    "Rtary2.Fast",  # 269
    "Rtary2.RiseTime",  # 270
    "Rtary2.Controller",  # 271
    # CLK
    "CLK.Multiply",  # 272
    "CLK.TimeBase",  # 273
    "CLK.Swing",  # 274
    # ARP
    "ARP.Direction",  # 275
    "ARP.Octaves",  # 276
    "ARP.Multiply",  # 277
    "ARP.Restart",  # 278
    "ARP.OnOff",  # 279
    "ARP.Order",  # 280
]


def build_name_to_index() -> Dict[str, int]:
    """Module-qualified name -> plugin parameter index."""
    return {name: index for index, name in enumerate(DIVA_PARAMETER_NAMES)}


def plugin_name(qualified_name: str) -> str:
    """The bare name the plugin reports for a qualified name ('VCF1.Model' -> 'Model')."""
    return qualified_name.split(".", 1)[1]


def module_name(qualified_name: str) -> str:
    """The owning module ('VCF1.Model' -> 'VCF1')."""
    return qualified_name.split(".", 1)[0]
