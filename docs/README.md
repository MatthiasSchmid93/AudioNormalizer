# ![plot](../assets/nor.png) AudioNormalizer
<p align="center">
 <img src="../assets/UI.png">
</p>
<br/>
<p align="center">
This is a normalizer for mp3 and wav files. Its a final project for Havard CS50P.

The approach in this program to normalize audio differs in comparison to most other programs that implement this task. Most normalizers amplify an audio signal  until the highest transient reaches 0 dB. The downside of this approach is that the audio signal can only be amplified based on a single maximum transient.     In other words, the whole normalizing process depends on one transient, which is inefficient.

This program, however, doesn't depend on a single transient. It rather splits the original signal into four separate signals (left positive, left negative,     right positive, and right negative). Then it uses an algorithm that finds several transients that exceed a calculated value based on the audio file's           duration and actual volume. For each split signal, a copy is being created, a dummy signal. In the dummy signals, the found transients are turned into the       value of zero. The next task is to find the new maximum transient in each modified dummy signal like any other simple normalizer. The new maximum value is       then saved for the final amplification.

At this stage, the program finds all amplitudes in the original split signals where the previously found transients are sitting. The split audio signals are     then amplified based on the new maximum transient. All amplitudes that exceed (clip) above 0 dB are lowered to 0 dB without manipulating the original signal     except for the volume.

Finally, all the amplified split signals are merged together and are saved as the original file name and type. All the ID3 tags, including the album cover,     are also maintained in the new normalized file.

As a result, the user gets an audiofile normalized to 0 dB without losing the dynamic range or the overall audio quality.
