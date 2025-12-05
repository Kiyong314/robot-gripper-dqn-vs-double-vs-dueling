--[[
=============================================================================
IRB360 Delta Robot Script with Python Control Support
=============================================================================

이 스크립트는 IRB360 델타 로봇을 제어합니다.
Python에서 ZMQ Remote API를 통해 moveToConfigFromPython 등의 함수를 호출할 수 있습니다.

주요 기능:
- IK/FK 모드 전환
- Python에서 호출 가능한 래퍼 함수들
- 초기 위치 설정
- sysCall_actuation에서 IK 동기화

사용법:
1. CoppeliaSim에서 irb360 오브젝트의 Script에 이 내용을 붙여넣기
2. 시뮬레이션 시작
3. Python에서 sim.callScriptFunction('moveToConfigFromPython', script_handle, goalConfig) 호출

좌표 시스템:
- goalConfig = {cartesianX, cartesianY, cartesianZ, motor_rotation}
- cartesianX/Y/Z: 로봇 베이스 기준 상대 좌표 (미터)
- motor_rotation: 회전 각도 (라디안)
=============================================================================
]]

sim = require'sim'
simIK = require'simIK'

-- 전역 변수 (Python 래퍼 함수에서 접근 가능하도록)
info = nil
ikEnv = nil

-- =============================================================================
-- FK/IK 모드 전환 함수
-- =============================================================================

function setFkMode(infoTable)
    if not infoTable then return end
    infoTable.ikMode = false
    simIK.setElementFlags(infoTable.ikEnv, infoTable.mainIkGroup, infoTable.platformIkElement, 0)
    simIK.setJointMode(infoTable.ikEnv, infoTable.fkDrivingJoints_inIkEnv[1], simIK.jointmode_passive)
    simIK.setJointMode(infoTable.ikEnv, infoTable.fkDrivingJoints_inIkEnv[2], simIK.jointmode_passive)
    simIK.setJointMode(infoTable.ikEnv, infoTable.fkDrivingJoints_inIkEnv[3], simIK.jointmode_passive)
end

function setIkMode(infoTable)
    if not infoTable then return end
    infoTable.ikMode = true
    simIK.setElementFlags(infoTable.ikEnv, infoTable.mainIkGroup, infoTable.platformIkElement, 1)
    simIK.setJointMode(infoTable.ikEnv, infoTable.fkDrivingJoints_inIkEnv[1], simIK.jointmode_ik)
    simIK.setJointMode(infoTable.ikEnv, infoTable.fkDrivingJoints_inIkEnv[2], simIK.jointmode_ik)
    simIK.setJointMode(infoTable.ikEnv, infoTable.fkDrivingJoints_inIkEnv[3], simIK.jointmode_ik)
end

-- =============================================================================
-- 이동 함수
-- =============================================================================

function moveToConfigCallback(data)
    local joints = data.auxData.fkDrivingJoints
    if data.auxData.ikMode then
        joints = data.auxData.ikDrivingJoints
    end
    for i = 1, #joints, 1 do
        sim.setJointPosition(joints[i], data.pos[i])
    end
    -- IK 계산은 sysCall_actuation에서 처리하므로 여기서는 생략
end

function moveToConfig(maxVelocity, maxAcceleration, maxJerk, goalConfig, infoTable)
    if not infoTable then return end
    local joints = infoTable.fkDrivingJoints
    if infoTable.ikMode then
        joints = infoTable.ikDrivingJoints
    end
    local startConfig = {}
    for i = 1, #joints, 1 do
        startConfig[i] = sim.getJointPosition(joints[i])
    end
    local params = {
        pos = startConfig,
        targetPos = goalConfig,
        maxVel = maxVelocity,
        maxAccel = maxAcceleration,
        maxJerk = maxJerk,
        callback = moveToConfigCallback,
        auxData = infoTable
    }
    sim.moveToConfig(params)
end

-- =============================================================================
-- Python에서 호출 가능한 래퍼 함수들
-- =============================================================================

function moveToConfigFromPython(goalConfig)
    if not info then
        sim.addLog(sim.verbosity_scripterrors, "IRB360: info not initialized")
        return false
    end
    setIkMode(info)
    
    local linVel = 2
    local linAccel = 3
    local linJerk = 30
    local angVel = 180 * math.pi / 180
    local angAccel = 360 * math.pi / 180
    local angJerk = 3600 * math.pi / 180
    
    local maxLinVel = {linVel, linVel, linVel, angVel}
    local maxLinAccel = {linAccel, linAccel, linAccel, angAccel}
    local maxLinJerk = {linJerk, linJerk, linJerk, angJerk}
    
    moveToConfig(maxLinVel, maxLinAccel, maxLinJerk, goalConfig, info)
    return true
end

function moveToConfigFastFromPython(goalConfig)
    if not info then return false end
    setIkMode(info)
    
    local linVel = 4
    local linAccel = 6
    local linJerk = 60
    local angVel = 360 * math.pi / 180
    local angAccel = 720 * math.pi / 180
    local angJerk = 7200 * math.pi / 180
    
    local maxLinVel = {linVel, linVel, linVel, angVel}
    local maxLinAccel = {linAccel, linAccel, linAccel, angAccel}
    local maxLinJerk = {linJerk, linJerk, linJerk, angJerk}
    
    moveToConfig(maxLinVel, maxLinAccel, maxLinJerk, goalConfig, info)
    return true
end

function moveToConfigSlowFromPython(goalConfig)
    if not info then return false end
    setIkMode(info)
    
    local linVel = 0.5
    local linAccel = 1
    local linJerk = 10
    local angVel = 45 * math.pi / 180
    local angAccel = 90 * math.pi / 180
    local angJerk = 900 * math.pi / 180
    
    local maxLinVel = {linVel, linVel, linVel, angVel}
    local maxLinAccel = {linAccel, linAccel, linAccel, angAccel}
    local maxLinJerk = {linJerk, linJerk, linJerk, angJerk}
    
    moveToConfig(maxLinVel, maxLinAccel, maxLinJerk, goalConfig, info)
    return true
end

function getIkTargetPositionForPython()
    local cartesianX = sim.getJointPosition(sim.getObject('../cartesianX'))
    local cartesianY = sim.getJointPosition(sim.getObject('../cartesianX/cartesianY'))
    local cartesianZ = sim.getJointPosition(sim.getObject('../cartesianX/cartesianY/cartesianZ'))
    local motor = sim.getJointPosition(sim.getObject('../motor'))
    return {cartesianX, cartesianY, cartesianZ, motor}
end

function setIkModeFromPython()
    if not info then return false end
    setIkMode(info)
    return true
end

function setFkModeFromPython()
    if not info then return false end
    setFkMode(info)
    return true
end

function goHomeFromPython()
    if not info then return false end
    setIkMode(info)
    
    local linVel = 2
    local linAccel = 3
    local linJerk = 30
    local angVel = 180 * math.pi / 180
    local angAccel = 360 * math.pi / 180
    local angJerk = 3600 * math.pi / 180
    
    local maxLinVel = {linVel, linVel, linVel, angVel}
    local maxLinAccel = {linAccel, linAccel, linAccel, angAccel}
    local maxLinJerk = {linJerk, linJerk, linJerk, angJerk}
    
    moveToConfig(maxLinVel, maxLinAccel, maxLinJerk, {0, 0, 0, 0}, info)
    return true
end

-- =============================================================================
-- 초기화 함수 (시뮬레이션 시작 시 가장 먼저 호출됨)
-- =============================================================================

function sysCall_init()
    -- FK 모드에서 제어하는 조인트들
    local fkDrivingJoints = {-1, -1, -1, -1}
    fkDrivingJoints[1] = sim.getObject('../drivingJoint1')
    fkDrivingJoints[2] = sim.getObject('../drivingJoint2')
    fkDrivingJoints[3] = sim.getObject('../drivingJoint3')
    fkDrivingJoints[4] = sim.getObject('../motor')
    
    -- IK 모드에서 제어하는 조인트들
    local ikDrivingJoints = {-1, -1, -1, -1}
    ikDrivingJoints[1] = sim.getObject('../cartesianX')
    ikDrivingJoints[2] = sim.getObject('../cartesianY')
    ikDrivingJoints[3] = sim.getObject('../cartesianZ')
    ikDrivingJoints[4] = sim.getObject('../motor')
    
    -- IK 환경 생성
    ikEnv = simIK.createEnvironment()
    
    -- 메인 IK 그룹
    local base = sim.getObject('..')
    local ikTip = sim.getObject('../ikTip')
    local ikTarget = sim.getObject('../ikTarget')
    local loop1Tip = sim.getObject('../loopTip')
    local loop1Target = sim.getObject('../loopTarget')
    local loop2Tip = sim.getObject('../loopTip0')
    local loop2Target = sim.getObject('../loopTarget0')
    local loop3Tip = sim.getObject('../loopTip1')
    local loop3Target = sim.getObject('../loopTarget1')
    local loop4Tip = sim.getObject('../loopTip2')
    local loop4Target = sim.getObject('../loopTarget2')
    local loop5Tip = sim.getObject('../loopTip3')
    local loop5Target = sim.getObject('../loopTarget3')
    
    local ikGroup_main = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup_main, simIK.method_pseudo_inverse, 0, 6)
    local ikElementTip = simIK.addElementFromScene(ikEnv, ikGroup_main, base, ikTip, ikTarget, simIK.constraint_position)
    local ikElement = simIK.addElementFromScene(ikEnv, ikGroup_main, base, loop1Tip, loop1Target, simIK.constraint_pose)
    ikElement = simIK.addElementFromScene(ikEnv, ikGroup_main, base, loop2Tip, loop2Target, simIK.constraint_pose)
    ikElement = simIK.addElementFromScene(ikEnv, ikGroup_main, base, loop3Tip, loop3Target, simIK.constraint_pose)
    ikElement = simIK.addElementFromScene(ikEnv, ikGroup_main, base, loop4Tip, loop4Target, simIK.constraint_pose)
    local ikElement2, mapping = simIK.addElementFromScene(ikEnv, ikGroup_main, base, loop5Tip, loop5Target, simIK.constraint_pose)
    local fkDrivingJoints_inIkEnv = {mapping[fkDrivingJoints[1]], mapping[fkDrivingJoints[2]], mapping[fkDrivingJoints[3]]}
    
    -- 센터 축 IK 그룹
    local axisBase = sim.getObject('../axisL')
    local axisTip = sim.getObject('../axisTip')
    local axisTarget = sim.getObject('../axisTarget')
    
    local ikGroup_axis = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup_axis, simIK.method_pseudo_inverse, 0, 6)
    ikElement = simIK.addElementFromScene(ikEnv, ikGroup_axis, axisBase, axisTip, axisTarget, simIK.constraint_pose)
    simIK.setElementPrecision(ikEnv, ikGroup_axis, ikElement, {0.0001, 0.1})
    
    -- 브리지 IK 그룹들
    local bridge1Base = sim.getObject('../j2')
    local bridge1Tip = sim.getObject('../bridgeLTip')
    local bridge1Target = sim.getObject('../bridgeLTarget')
    local ikGroup_bridge1 = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup_bridge1, simIK.method_damped_least_squares, 0.01, 3)
    simIK.addElementFromScene(ikEnv, ikGroup_bridge1, bridge1Base, bridge1Tip, bridge1Target, simIK.constraint_position)
    
    local bridge2Base = sim.getObject('../j26')
    local bridge2Tip = sim.getObject('../bridgeRTip')
    local bridge2Target = sim.getObject('../bridgeRTarget')
    local ikGroup_bridge2 = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup_bridge2, simIK.method_damped_least_squares, 0.01, 3)
    simIK.addElementFromScene(ikEnv, ikGroup_bridge2, bridge2Base, bridge2Tip, bridge2Target, simIK.constraint_position)
    
    local bridge3Base = sim.getObject('../j28')
    local bridge3Tip = sim.getObject('../bridgeLTip0')
    local bridge3Target = sim.getObject('../bridgeLTarget0')
    local ikGroup_bridge3 = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup_bridge3, simIK.method_damped_least_squares, 0.01, 3)
    simIK.addElementFromScene(ikEnv, ikGroup_bridge3, bridge3Base, bridge3Tip, bridge3Target, simIK.constraint_position)
    
    local bridge4Base = sim.getObject('../j29')
    local bridge4Tip = sim.getObject('../bridgeRTip0')
    local bridge4Target = sim.getObject('../bridgeRTarget0')
    local ikGroup_bridge4 = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup_bridge4, simIK.method_damped_least_squares, 0.01, 3)
    simIK.addElementFromScene(ikEnv, ikGroup_bridge4, bridge4Base, bridge4Tip, bridge4Target, simIK.constraint_position)
    
    local bridge5Base = sim.getObject('../j31')
    local bridge5Tip = sim.getObject('../bridgeLTip1')
    local bridge5Target = sim.getObject('../bridgeLTarget1')
    local ikGroup_bridge5 = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup_bridge5, simIK.method_damped_least_squares, 0.01, 3)
    simIK.addElementFromScene(ikEnv, ikGroup_bridge5, bridge5Base, bridge5Tip, bridge5Target, simIK.constraint_position)
    
    local bridge6Base = sim.getObject('../j34')
    local bridge6Tip = sim.getObject('../bridgeRTip1')
    local bridge6Target = sim.getObject('../bridgeRTarget1')
    local ikGroup_bridge6 = simIK.createGroup(ikEnv)
    simIK.setGroupCalculation(ikEnv, ikGroup_bridge6, simIK.method_damped_least_squares, 0.01, 3)
    simIK.addElementFromScene(ikEnv, ikGroup_bridge6, bridge6Base, bridge6Tip, bridge6Target, simIK.constraint_position)
    
    -- 전역 info 테이블 설정
    info = {}
    info.robotHandle = base
    info.ikMode = true  -- 기본값을 IK 모드로 설정
    info.ikEnv = ikEnv
    info.mainIkGroup = ikGroup_main
    info.platformIkElement = ikElementTip
    info.axisIkGroup = ikGroup_axis
    info.bridgeIkGroups = {ikGroup_bridge1, ikGroup_bridge2, ikGroup_bridge3, ikGroup_bridge4, ikGroup_bridge5, ikGroup_bridge6}
    info.fkDrivingJoints = fkDrivingJoints
    info.ikDrivingJoints = ikDrivingJoints
    info.fkDrivingJoints_inIkEnv = fkDrivingJoints_inIkEnv
    
    -- IK 모드 설정
    setIkMode(info)
    
    -- 초기 안전 위치로 이동 (높은 Z 위치)
    -- 홈 위치: X=0.014, Y=0.026, Z=-0.11 (로봇 베이스 기준 상대 좌표)
    -- 이는 월드 좌표 (0, 0, 0.6)에 해당하며, 작업 영역 밖 안전한 위치
    local homeConfig = {0.264, 0.026, -0.5, 0}  -- {cartesianX, cartesianY, cartesianZ, motor}
    
    -- 조인트 위치 직접 설정 (moveToConfig 대신 즉시 설정)
    sim.setJointPosition(ikDrivingJoints[1], homeConfig[1])  -- cartesianX
    sim.setJointPosition(ikDrivingJoints[2], homeConfig[2])  -- cartesianY
    sim.setJointPosition(ikDrivingJoints[3], homeConfig[3])  -- cartesianZ
    sim.setJointPosition(ikDrivingJoints[4], homeConfig[4])  -- motor
    
    sim.addLog(sim.verbosity_scriptinfos, "IRB360 initialized at safe home position")
end

-- =============================================================================
-- 매 시뮬레이션 스텝마다 호출되는 함수 (IK 동기화)
-- =============================================================================

function sysCall_actuation()
    if info and info.ikEnv then
        -- 메인 IK 그룹 처리
        local result1 = simIK.handleGroup(info.ikEnv, info.mainIkGroup, {syncWorlds = true})
        -- 축 IK 그룹 처리
        local result2 = simIK.handleGroup(info.ikEnv, info.axisIkGroup, {syncWorlds = true})
        
        -- 성공 시 브리지 IK 그룹 처리
        if result1 == simIK.result_success and result2 == simIK.result_success then
            for i = 1, #info.bridgeIkGroups, 1 do
                simIK.handleGroup(info.ikEnv, info.bridgeIkGroups[i], {syncWorlds = true, allowError = true})
            end
        end
        -- 참고: IK 실패 로그는 너무 많이 출력되므로 비활성화
    end
end

-- =============================================================================
-- 정리 함수 (시뮬레이션 종료 시 호출됨)
-- =============================================================================

function sysCall_cleanup()
    if ikEnv then
        simIK.eraseEnvironment(ikEnv)
    end
end
