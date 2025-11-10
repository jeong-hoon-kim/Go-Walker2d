import sys

# 계단 시작 x 좌표
start_x = 2.0 
# 계단 기본 z 좌표 (바닥)
start_z = 0.0

# 요청하신 계단 규격
stair_height = 0.18
stair_depth = 0.3
stair_count = 50

# geom의 y축 반폭 (좌우 폭)
y_half_width = 5.0 

# MuJoCo geom은 중심 좌표와 반폭(half-width/height/depth)을 사용합니다.
half_height = stair_height / 2.0
half_depth = stair_depth / 2.0

xml_output = []

# 첫 번째 계단의 바닥 좌표
current_x_edge = start_x
current_z_base = start_z

for i in range(1, stair_count + 1):
    # 계단 블록의 중심(center) 좌표 계산
    center_x = current_x_edge + half_depth
    center_z = current_z_base + half_height
    
    # <geom> 태그 속성값
    # pos: 중심 좌표 (x, y, z)
    # size: 반폭 (x, y, z)
    pos_str = f"{center_x:.2f} 0 {center_z:.2f}"
    size_str = f"{half_depth:.2f} {y_half_width} {half_height:.2f}"
    name_str = f"stair_{i}"
    
    # XML 라인 생성
    line = f'    <geom conaffinity="1" condim="3" name="{name_str}" pos="{pos_str}" size="{size_str}" type="box" rgba="0.9 0.5 0.5 1"/>'
    xml_output.append(line)
    
    # 다음 계단의 시작 지점(바닥) 업데이트
    current_x_edge += stair_depth
    current_z_base += stair_height

# 생성된 모든 XML 라인을 출력
print("\n".join(xml_output))